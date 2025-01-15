import os
import pandas as pd
import numpy as np
import re
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Load CSV files from a directory based on a filename pattern
def load_csv_files(directory, filename_pattern):
    files_data = {}
    for filename in os.listdir(directory):
        if filename_pattern in filename:
            df = pd.read_csv(os.path.join(directory, filename))
            df.dropna(inplace=True)
            base_name = filename.split(filename_pattern)[0]
            files_data[base_name] = df
    return files_data

# Load features from .npy files based on a filename pattern
def load_features(directory, filename_pattern):
    files_data = {}
    for filename in os.listdir(directory):
        if filename_pattern in filename:
            features = pd.read_csv(os.path.join(directory, filename))
            features.dropna(inplace=True)
            base_name = filename.split('_features')[0]
            files_data[base_name] = features
    return files_data

# Find corresponding phoneme files based on the base names of position filenames
def find_phoneme_files(directory, base_names):
    phoneme_files = {}
    for base_name in base_names:
        phoneme_file = os.path.join(directory, f'{base_name}.lpc')
        if os.path.exists(phoneme_file):
            phoneme_files[base_name] = phoneme_file
    return phoneme_files


import os
import re
import csv
import cv2
import numpy as np
import torch
import pandas as pd
import librosa
from praatio import textgrid as tgio
from tqdm import tqdm

# ==========================================================
# Helper Functions
# ==========================================================

def pad_sequences(sequences, max_length, pad_value=0):
    """
    Pad sequences to the maximum length.

    Args:
        sequences (list): List of sequences to pad.
        max_length (int): Maximum length to pad to.
        pad_value (int): Value to use for padding.

    Returns:
        np.ndarray: Padded sequences.
    """
    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_length:
            padding = np.full((max_length - len(seq), seq.shape[1]), pad_value)
            padded_seq = np.vstack((seq, padding))
        else:
            padded_seq = seq[:max_length]
        padded_sequences.append(padded_seq)
    return np.array(padded_sequences)

def combine_sequences_with_padding(video_data):
    """
    Combine sequences with padding to ensure uniform length.

    Args:
        video_data (dict): Dictionary containing video data.

    Returns:
        tuple: Padded input sequences (X_student_hand_shape, X_student_hand_pos, X_student_lips, X_teacher) and padded labels (y).
    """
    max_length = max(len(video_data[video]["X_student_hand_shape"]) for video in video_data)
    
    # Pad hand shape features
    X_student_hand_shape_padded = [
        pad_sequences([video_data[video]["X_student_hand_shape"]], max_length)[0] for video in video_data
    ]
    
    # Pad hand position features
    X_student_hand_pos_padded = [
        pad_sequences([video_data[video]["X_student_hand_pos"]], max_length)[0] for video in video_data
    ]
    
    # Pad lip features
    X_student_lips_padded = [
        pad_sequences([video_data[video]["X_student_lips"]], max_length)[0] for video in video_data
    ]
    
    # Pad teacher features
    X_teacher_padded = [
        pad_sequences([video_data[video]["X_teacher"]], max_length)[0] for video in video_data
    ]
    
    # Pad labels
    y_padded = [
        video_data[video]["y"]
        + [phoneme_to_index[" "]] * (max_length - len(video_data[video]["y"]))
        for video in video_data
    ]
    
    return X_student_hand_shape_padded, X_student_hand_pos_padded, X_student_lips_padded, X_teacher_padded, y_padded

def compute_log_mel_spectrogram(audio_path, sr=16000, n_fft=400, hop_length=160, n_mels=80):
    """
    Compute the log-mel spectrogram for an audio file.

    Args:
        audio_path (str): Path to the audio file.
        sr (int): Sample rate.
        n_fft (int): FFT window size.
        hop_length (int): Hop length for STFT.
        n_mels (int): Number of mel bands.

    Returns:
        np.ndarray: Log-mel spectrogram of shape (num_frames, n_mels).
    """
    # Load audio
    audio, _ = librosa.load(audio_path, sr=sr)

    # Compute mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )

    # Convert to log scale
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Transpose to (num_frames, n_mels)
    log_mel_spectrogram = log_mel_spectrogram.T

    return log_mel_spectrogram

def parse_textgrid(textgrid_path):
    """
    Parse a TextGrid file to extract phoneme-level intervals.

    Args:
        textgrid_path (str): Path to the TextGrid file.

    Returns:
        list: List of (start_time, end_time, phoneme) tuples.
    """
    tg = tgio.openTextgrid(textgrid_path, includeEmptyIntervals=False)
    phone_tier = tg.getTier("phones")
    return [(start, end, label) for start, end, label in phone_tier.entries]

def get_phoneme_labels_for_frames(phoneme_intervals, num_frames, fps):
    """
    Map phoneme intervals to video frames.

    Args:
        phoneme_intervals (list): List of (start_time, end_time, phoneme) tuples.
        num_frames (int): Total number of video frames.
        fps (int): Frame rate of the video.

    Returns:
        list: Phoneme labels for each frame.
    """
    phoneme_labels = []
    for frame_idx in range(num_frames):
        frame_time = frame_idx / fps
        phoneme = " "  # Default to silence/space
        for start, end, label in phoneme_intervals:
            if start <= frame_time < end:
                phoneme = label
                break
        phoneme_labels.append(phoneme)
    return phoneme_labels

# Load phoneme-to-index mapping
with open(
    r"/scratch2/bsow/Documents/ACSR/data/training_videos/phoneme_dictionary.txt", "r"
) as file:
    reader = csv.reader(file)
    vocabulary_list = [row[0] for row in reader]

phoneme_to_index = {phoneme: idx for idx, phoneme in enumerate(vocabulary_list)}
index_to_phoneme = {idx: phoneme for phoneme, idx in phoneme_to_index.items()}
phoneme_to_index[" "] = len(phoneme_to_index)
index_to_phoneme[len(index_to_phoneme)] = " "

def load_coordinates(directory, base_name):
    """
    Load pre-extracted coordinates from a CSV file.

    Args:
        directory (str): Directory containing the coordinate files.
        base_name (str): Base name of the video (e.g., 'sent_01').

    Returns:
        pd.DataFrame: DataFrame containing the coordinates.
    """
    file_path = os.path.join(directory, f"{base_name}_coordinates.csv")
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)  # Drop rows with NaN values
    return df

def prepare_data_for_videos_no_sliding_windows(
    base_names, phoneme_files, audio_dir, textgrid_dir, video_dir, coordinates_dir
):
    """
    Prepare data for all videos without sliding windows.

    Args:
        base_names (list): List of base names for videos.
        phoneme_files (dict): Dictionary of phoneme file paths.
        audio_dir (str): Directory containing audio files.
        textgrid_dir (str): Directory containing TextGrid files.
        video_dir (str): Directory containing video files.
        coordinates_dir (str): Directory containing pre-extracted coordinate files.

    Returns:
        dict: Dictionary containing combined features, spectrograms, and phoneme sequences.
    """
    all_videos_data = {}
    for base_name in base_names:
        if base_name in phoneme_files:
            # Load pre-extracted coordinates
            coordinates_df = load_coordinates(coordinates_dir, base_name)
            if 'frame_number' not in coordinates_df.columns:
                raise ValueError(f"Coordinate file for {base_name} does not contain 'frame_number' column.")
            frame_numbers = coordinates_df['frame_number'].values

            # Separate coordinates into hand shape, hand position, and lip landmarks
            hand_shape_columns = [f"hand_x{i}" for i in range(21)] + [f"hand_y{i}" for i in range(21)] + [f"hand_z{i}" for i in range(21)]
            hand_pos_columns = ["hand_pos_x", "hand_pos_y", "hand_pos_z"]
            lip_columns = [f"lip_x{i}" for i in range(40)] + [f"lip_y{i}" for i in range(40)] + [f"lip_z{i}" for i in range(40)]

            X_student_hand_shape = coordinates_df[hand_shape_columns].to_numpy()
            X_student_hand_pos = coordinates_df[hand_pos_columns].to_numpy()
            X_student_lips = coordinates_df[lip_columns].to_numpy()

            # Load audio and compute spectrogram
            audio_path = os.path.join(audio_dir, f"{base_name}.wav")
            log_mel_spectrogram = compute_log_mel_spectrogram(audio_path)

            # Load TextGrid and get phoneme labels for the entire sequence
            textgrid_path = os.path.join(textgrid_dir, f"{base_name}.TextGrid")
            phoneme_intervals = parse_textgrid(textgrid_path)

            # Extract phoneme labels as a sequence (without frame-level alignment)
            phoneme_labels = [interval[2] for interval in phoneme_intervals]  # Extract phoneme labels

            # Convert phoneme labels to indices
            phoneme_indices = [phoneme_to_index.get(phoneme, -1) for phoneme in phoneme_labels]

            # Combine features, spectrogram, and phoneme indices
            all_videos_data[base_name] = {
                "X_student_hand_shape": X_student_hand_shape,  # Hand shape coordinates
                "X_student_hand_pos": X_student_hand_pos,      # Hand position coordinates
                "X_student_lips": X_student_lips,              # Lip landmarks
                "X_teacher": log_mel_spectrogram,              # Audio features (log-mel spectrogram)
                "y": phoneme_indices,                          # Phoneme labels (sequence)
            }
    return all_videos_data


# Directories
data_dir = r'/scratch2/bsow/Documents/ACSR/output/predictions'
phoneme_dir = r'/scratch2/bsow/Documents/ACSR/data/training_videos/lpc'
audio_dir = r'/scratch2/bsow/Documents/ACSR/data/training_videos/audio'
textgrid_dir = r'/scratch2/bsow/Documents/ACSR/data/training_videos/textgrids'
video_dir = r'/scratch2/bsow/Documents/ACSR/data/training_videos/videos'
coordinates_dir = r'/scratch2/bsow/Documents/ACSR/output/extracted_coordinates'

coordinates_data = load_csv_files(coordinates_dir, "_coordinates")
# Find phoneme files
base_names = coordinates_data.keys()
phoneme_files = find_phoneme_files(phoneme_dir, base_names)

# Prepare data
all_videos_data = prepare_data_for_videos_no_sliding_windows(
    base_names, phoneme_files, audio_dir, textgrid_dir, video_dir, coordinates_dir
)

# Combine sequences with padding
X_student_hand_shape_padded, X_student_hand_pos_padded, X_student_lips_padded, X_teacher_padded, y_padded = combine_sequences_with_padding(all_videos_data)

# Convert to PyTorch tensors
X_student_hand_shape_tensor = torch.tensor(X_student_hand_shape_padded, dtype=torch.float32)
X_student_hand_pos_tensor = torch.tensor(X_student_hand_pos_padded, dtype=torch.float32)
X_student_lips_tensor = torch.tensor(X_student_lips_padded, dtype=torch.float32)
X_teacher_tensor = torch.tensor(X_teacher_padded, dtype=torch.float32)
y_tensor = torch.tensor(y_padded, dtype=torch.long)

# Final organized data
all_videos_data = {
    "X_student_hand_shape": X_student_hand_shape_tensor,  # Hand shape coordinates
    "X_student_hand_pos": X_student_hand_pos_tensor,      # Hand position coordinates
    "X_student_lips": X_student_lips_tensor,              # Lip landmarks
    "X_teacher": X_teacher_tensor,                        # Audio features (log-mel spectrogram)
    "y": y_tensor,                                        # Phoneme labels
}


import torch
from torch.utils.data import DataLoader, TensorDataset

# Function to split data into training and validation sets
def train_val_split(data, train_ratio=0.9):
    """
    Split data into training and validation sets.

    Args:
        data (dict): Dictionary containing the dataset.
        train_ratio (float): Proportion of data to use for training.

    Returns:
        tuple: Two dictionaries for training and validation data.
    """
    num_samples = len(data['X_student_hand_shape'])
    split_idx = int(num_samples * train_ratio)
    
    # Randomize the data
    indices = torch.randperm(num_samples)
    
    # Split hand shape features
    X_student_hand_shape = data['X_student_hand_shape'][indices]
    X_student_hand_shape_train = X_student_hand_shape[:split_idx]
    X_student_hand_shape_val = X_student_hand_shape[split_idx:]
    
    # Split hand position features
    X_student_hand_pos = data['X_student_hand_pos'][indices]
    X_student_hand_pos_train = X_student_hand_pos[:split_idx]
    X_student_hand_pos_val = X_student_hand_pos[split_idx:]
    
    # Split lip features
    X_student_lips = data['X_student_lips'][indices]
    X_student_lips_train = X_student_lips[:split_idx]
    X_student_lips_val = X_student_lips[split_idx:]
    
    # Split teacher features
    X_teacher = data['X_teacher'][indices]
    X_teacher_train = X_teacher[:split_idx]
    X_teacher_val = X_teacher[split_idx:]
    
    # Split labels
    y = data['y'][indices]
    y_train = y[:split_idx]
    y_val = y[split_idx:]
    
    # Create train and validation data dictionaries
    train_data = {
        'X_student_hand_shape': X_student_hand_shape_train,
        'X_student_hand_pos': X_student_hand_pos_train,
        'X_student_lips': X_student_lips_train,
        'X_teacher': X_teacher_train,
        'y': y_train
    }
    val_data = {
        'X_student_hand_shape': X_student_hand_shape_val,
        'X_student_hand_pos': X_student_hand_pos_val,
        'X_student_lips': X_student_lips_val,
        'X_teacher': X_teacher_val,
        'y': y_val
    }
    
    return train_data, val_data


# Convert data to DataLoader format
def data_to_dataloader(data, batch_size=4, shuffle=True):
    """
    Convert data into PyTorch DataLoader format.

    Args:
        data (dict): Dictionary containing the dataset.
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: PyTorch DataLoader object.
    """
    X_student_hand_shape_tensors = data['X_student_hand_shape']
    X_student_hand_pos_tensors = data['X_student_hand_pos']
    X_student_lips_tensors = data['X_student_lips']
    X_teacher_tensors = data['X_teacher']
    y_tensors = data['y']
    
    # Create a TensorDataset with inputs and labels
    dataset = TensorDataset(
        X_student_hand_shape_tensors,
        X_student_hand_pos_tensors,
        X_student_lips_tensors,
        X_teacher_tensors,
        y_tensors
    )
    
    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


# Split data
train_data, val_data = train_val_split(all_videos_data)

# Prepare DataLoaders
train_loader = data_to_dataloader(train_data, batch_size=4, shuffle=True)
val_loader = data_to_dataloader(val_data, batch_size=4, shuffle=False)

print("Len of train dataset", len(train_data['X_student_hand_shape']))
print("Len of val dataset", len(val_data['X_student_hand_shape']))

# Check the DataLoader output
for batch_X_student_hand_shape, batch_X_student_hand_pos, batch_X_student_lips, batch_X_teacher, batch_y in train_loader:
    print("Batch X_student_hand_shape shape:", batch_X_student_hand_shape.shape)
    print("Batch X_student_hand_pos shape:", batch_X_student_hand_pos.shape)
    print("Batch X_student_lips shape:", batch_X_student_lips.shape)
    print("Batch X_teacher shape:", batch_X_teacher.shape)
    print("Batch y shape:", batch_y.shape)
    break


import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import pandas as pd

class IPADataset(Dataset):
    def __init__(self, manifest_file, alphabet_file, sample_rate=16000, n_mels=80):
        self.manifest = pd.read_csv(manifest_file, header=None)
        with open(alphabet_file, "r") as f:
            self.alphabet = f.read().splitlines()
        self.sample_rate = sample_rate
        self.n_mels = n_mels

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        audio_path, ipa_path = self.manifest.iloc[idx]
        
        # Load audio and compute mel spectrogram
        audio, _ = librosa.load(audio_path, sr=self.sample_rate)
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.sample_rate, n_mels=self.n_mels
        )
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Load IPA transcription
        with open(ipa_path, "r") as f:
            ipa = f.read().strip()
        
        # Convert IPA to indices
        ipa_indices = [self.alphabet.index(c) for c in ipa.split()]
        
        return torch.tensor(mel_spec, dtype=torch.float32), torch.tensor(ipa_indices, dtype=torch.long)
    

class DeepSpeech2(nn.Module):
    def __init__(self, num_classes, hidden_size=1024, num_layers=5, n_mels=80):
        super(DeepSpeech2, self).__init__()
        
        # 2D Convolutional Layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        # Bidirectional LSTM Layers
        self.lstm = nn.LSTM(
            input_size=32 * n_mels,  # Output size of the last convolutional layer
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
        )
        
        # Fully Connected Layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional LSTM
        
    def forward(self, x):
        
        # Convolutional Layers
        x = F.relu(self.conv1(x))
        
        x = F.relu(self.conv2(x))
        
        # Reshape for LSTM
        batch_size, channels, n_mels, time_steps = x.size()
        x = x.permute(0, 3, 1, 2)  # Move time_steps to the second dimension
        x = x.reshape(batch_size, time_steps, -1)  # Flatten the last two dimensions
        
        # LSTM Layers
        x, _ = self.lstm(x)
        
        # Fully Connected Layer
        x = self.fc(x)
        
        return x
    
import torch
import torch.nn as nn

class StudentModel(nn.Module):
    def __init__(self, hand_shape_dim, hand_pos_dim, lips_dim, hidden_dim, output_dim):
        super(StudentModel, self).__init__()
        
        # Feature extractors
        self.hand_shape_extractor = nn.Sequential(
            nn.Linear(hand_shape_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        self.hand_pos_extractor = nn.Sequential(
            nn.Linear(hand_pos_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        self.lips_extractor = nn.Sequential(
            nn.Linear(lips_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # BiLSTM layers
        self.bilstm = nn.LSTM(
            input_size=64 + 32 + 128,  # Combined feature size
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # Output phoneme predictions

    def forward(self, hand_shape, hand_pos, lips):
        # Extract features
        hand_shape_features = self.hand_shape_extractor(hand_shape)
        hand_pos_features = self.hand_pos_extractor(hand_pos)
        lips_features = self.lips_extractor(lips)
        
        # Combine features
        combined_features = torch.cat([hand_shape_features, hand_pos_features, lips_features], dim=-1)
        
        # Pass through BiLSTM
        lstm_out, _ = self.bilstm(combined_features)
        
        # Final predictions
        output = self.fc(lstm_out)
        return output

import torch.nn.functional as F

def sequence_level_distillation_loss(student_logits, teacher_logits, batch_y, input_lengths, label_lengths, device):
    """
    Compute the sequence-level distillation loss.

    Args:
        student_logits (torch.Tensor): Logits from the student model.
        teacher_logits (torch.Tensor): Logits from the teacher model.
        batch_y (torch.Tensor): Padded ground-truth labels.
        input_lengths (torch.Tensor): Lengths of input sequences.
        label_lengths (torch.Tensor): Lengths of label sequences (excluding padding).

    Returns:
        torch.Tensor: Combined loss value.
    """
    # Move tensors to the device
    student_logits = student_logits.to(device)
    teacher_logits = teacher_logits.to(device)
    batch_y = batch_y.to(device)
    input_lengths = input_lengths.to(device)
    label_lengths = label_lengths.to(device)
    
    # Cosine similarity loss
    cosine_loss = 1 - F.cosine_similarity(student_logits, teacher_logits, dim=-1).mean()
    
    # CTC loss
    log_probs = F.log_softmax(student_logits, dim=-1)  # Log-softmax of student_logits
    log_probs = log_probs.permute(1, 0, 2)  # Reshape to [sequence_length, batch_size, num_classes]
    
    ctc_loss = F.ctc_loss(
        log_probs,
        batch_y,  # Padded labels
        input_lengths,
        label_lengths,  # Lengths of label sequences (excluding padding)
        blank=phoneme_to_index[" "],  # Blank token index
        reduction='mean'
    )
    
    # Combine losses
    total_loss = cosine_loss + ctc_loss
    return total_loss

from torch.optim import Adam
from torch.utils.data import DataLoader

# Initialize the student model
student_model = StudentModel(
    hand_shape_dim=63,  # 21 keypoints × 3 coordinates
    hand_pos_dim=3,     # 3 coordinates (x, y, z)
    lips_dim=120,       # 40 keypoints × 3 coordinates
    hidden_dim=256,
    output_dim=len(phoneme_to_index)  # Number of phonemes
)

# Load the pretrained teacher model
teacher_model = DeepSpeech2(num_classes=len(phoneme_to_index))
teacher_model.load_state_dict(torch.load("/scratch2/bsow/Documents/ACSR/output/saved_models/deepspeech2_pretrained.pth"))

# Set teacher model to evaluation mode
teacher_model.eval()

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on device:", device)
student_model.to(device)
teacher_model.to(device)

# Optimizer
optimizer = Adam(student_model.parameters(), lr=1e-3)

def train_student_model(student_model, teacher_model, train_loader, num_epochs=50):
    student_model.train()
    teacher_model.eval()  # Set teacher model to evaluation mode
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for batch_X_student_hand_shape, batch_X_student_hand_pos, batch_X_student_lips, batch_X_teacher, batch_y in train_loader:
            # Move data to device
            batch_X_student_hand_shape = batch_X_student_hand_shape.to(device)
            batch_X_student_hand_pos = batch_X_student_hand_pos.to(device)
            batch_X_student_lips = batch_X_student_lips.to(device)
            batch_X_teacher = batch_X_teacher.to(device)
            batch_y = batch_y.to(device)

            # Reshape batch_X_teacher for the teacher model
            batch_X_teacher = batch_X_teacher.unsqueeze(1)  # Add channel dimension
            batch_X_teacher = batch_X_teacher.permute(0, 1, 3, 2)  # Transpose to (batch_size, 1, num_mel_bins, time_steps)
            
            # Forward pass through the student model
            student_logits = student_model(batch_X_student_hand_shape, batch_X_student_hand_pos, batch_X_student_lips)
            
            # Forward pass through the teacher model
            with torch.no_grad():
                teacher_logits = teacher_model(batch_X_teacher)  # Ensure teacher model outputs logits
            
            # Compute input_lengths
            input_lengths = torch.full(
                (batch_X_student_hand_shape.size(0),),  # Batch size
                student_logits.size(1),  # Sequence length (time steps) from student_logits
                dtype=torch.long,
                device=device
            )
            
            # Compute label_lengths (excluding padding)
            label_lengths = torch.tensor(
                [len(seq[seq != phoneme_to_index[" "]]) for seq in batch_y],  # Length of each label sequence (excluding padding)
                dtype=torch.long,
                device=device
            )
            
            # Compute loss
            loss = sequence_level_distillation_loss(student_logits, teacher_logits, batch_y, input_lengths, label_lengths, device)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader)}")
    
    print("Training complete.")

# Start training
train_student_model(student_model, teacher_model, train_loader, num_epochs=50)


def evaluate_student_model(student_model, val_loader):
    student_model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch_X_student_hand_shape, batch_X_student_hand_pos, batch_X_student_lips, batch_X_teacher, batch_y in val_loader:
            # Move data to device
            batch_X_student_hand_shape = batch_X_student_hand_shape.to(device)
            batch_X_student_hand_pos = batch_X_student_hand_pos.to(device)
            batch_X_student_lips = batch_X_student_lips.to(device)
            batch_X_teacher = batch_X_teacher.to(device)
            batch_y = batch_y.to(device)
            
            # Reshape batch_X_teacher for the teacher model
            batch_X_teacher = batch_X_teacher.unsqueeze(1)  # Add channel dimension
            batch_X_teacher = batch_X_teacher.permute(0, 1, 3, 2)  # Transpose to (batch_size, 1, num_mel_bins, time_steps)
            
            # Forward pass through the student model
            student_logits = student_model(batch_X_student_hand_shape, batch_X_student_hand_pos, batch_X_student_lips)
            
            # Forward pass through the teacher model
            teacher_logits = teacher_model(batch_X_teacher)  # Ensure teacher model outputs logits
            
            # Compute input_lengths
            input_lengths = torch.full(
                (batch_X_student_hand_shape.size(0),),  # Batch size
                student_logits.size(1),  # Sequence length (time steps) from student_logits
                dtype=torch.long,
                device=device
            )
            
            # Compute label_lengths (excluding padding)
            label_lengths = torch.tensor(
                [len(seq[seq != 42]) for seq in batch_y],  # Length of each label sequence (excluding padding)
                dtype=torch.long,
                device=device
            )
            
            # Compute loss
            loss = sequence_level_distillation_loss(student_logits, teacher_logits, batch_y, input_lengths, label_lengths, device)
            total_loss += loss.item()
    
    print(f"Validation Loss: {total_loss / len(val_loader)}")
        
# Start training
train_student_model(student_model, teacher_model, train_loader, num_epochs=10)
evaluate_student_model(student_model, val_loader)

# Save the trained student model
torch.save(student_model.state_dict(), "/scratch2/bsow/Documents/ACSR/output/saved_models/student_model.pth")
print("Student model saved.")