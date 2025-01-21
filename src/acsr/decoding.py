import os
import pandas as pd
import numpy as np
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.optim import Adam
import librosa
from praatio import textgrid as tgio
from tqdm import tqdm

# Load CSV files from a directory based on a filename pattern
def load_csv_files(directory, filename_pattern):
    files_data = {}
    for filename in os.listdir(directory):
        if filename_pattern in filename:
            df = pd.read_csv(os.path.join(directory, filename))
            df.dropna(inplace=True)
            base_name = filename.split(filename_pattern)[0]
            if "sent_" in base_name:
                continue
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
        phoneme_file = os.path.join(directory, f'{base_name}.csv')
        if os.path.exists(phoneme_file):
            phoneme_files[base_name] = phoneme_file
    return phoneme_files


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
    #X_teacher_padded = [
    #    pad_sequences([video_data[video]["X_teacher"]], max_length)[0] for video in video_data
    #]
    
    # Pad labels
    y_padded = [
        video_data[video]["y"]
        + [phoneme_to_index[" "]] * (max_length - len(video_data[video]["y"]))
        for video in video_data
    ]
    
    return X_student_hand_shape_padded, X_student_hand_pos_padded, X_student_lips_padded, y_padded

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
    r"/scratch2/bsow/Documents/ACSR/data/training_videos/CSF22_train/phonelist.csv", "r"
) as file:
    reader = csv.reader(file)
    vocabulary_list = [row[0] for row in reader]

phoneme_to_index = {phoneme: idx for idx, phoneme in enumerate(set(list(vocabulary_list)))}
phoneme_to_index[" "] = len(phoneme_to_index)
index_to_phoneme = {idx: phoneme for phoneme, idx in phoneme_to_index.items()}

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
    base_names, phoneme_files, audio_dir, textgrid_dir, video_dir, coordinates_dir, labels_dir, phoneme_to_index
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
        labels_dir (str): Directory containing phoneme label CSV files.
        phoneme_to_index (dict): Mapping of phonemes to indices.

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

            # Separate coordinates into hand shape, hand position, and lip landmarks
            hand_shape_columns = [f"hand_x{i}" for i in range(21)] + [f"hand_y{i}" for i in range(21)] + [f"hand_z{i}" for i in range(21)]
            hand_pos_columns = hand_shape_columns
            lip_columns = [f"lip_x{i}" for i in range(39)] + [f"lip_y{i}" for i in range(39)] + [f"lip_z{i}" for i in range(39)]

            X_student_hand_shape = coordinates_df[hand_shape_columns].to_numpy()
            X_student_hand_pos = coordinates_df[hand_pos_columns].to_numpy()
            X_student_lips = coordinates_df[lip_columns].to_numpy()

            # Load audio and compute spectrogram
            #audio_path = os.path.join(audio_dir, f"{base_name}.wav")
            #log_mel_spectrogram = compute_log_mel_spectrogram(audio_path)

            # Load phoneme labels from CSV file
            labels_path = os.path.join(labels_dir, f"{base_name}.csv")
            if not os.path.exists(labels_path):
                raise FileNotFoundError(f"Phoneme label file not found: {labels_path}")
            
            # Read the CSV file
            phoneme_labels = pd.read_csv(labels_path, header=None).squeeze().tolist()  # Convert to list of phonemes

            # Convert phoneme labels to indices
            phoneme_indices = []
            for phoneme in phoneme_labels:
                if phoneme not in phoneme_to_index:
                    raise ValueError(f"Phoneme '{phoneme}' not found in the vocabulary. File: {base_name}")
                phoneme_indices.append(phoneme_to_index[phoneme])

            # Combine features, spectrogram, and phoneme indices
            all_videos_data[base_name] = {
                "X_student_hand_shape": X_student_hand_shape,  # Hand shape coordinates
                "X_student_hand_pos": X_student_hand_pos,      # Hand position coordinates
                "X_student_lips": X_student_lips,              # Lip landmarks
                #"X_teacher": log_mel_spectrogram,              # Audio features (log-mel spectrogram)
                "y": phoneme_indices,                          # Phoneme labels (sequence)
            }
    return all_videos_data


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
    #X_teacher = data['X_teacher'][indices]
    #X_teacher_train = X_teacher[:split_idx]
    #X_teacher_val = X_teacher[split_idx:]
    
    # Split labels
    y = data['y'][indices]
    y_train = y[:split_idx]
    print("Length of y_train: ", len(y_train))
    y_val = y[split_idx:]
    
    # Create train and validation data dictionaries
    train_data = {
        'X_student_hand_shape': X_student_hand_shape_train,
        'X_student_hand_pos': X_student_hand_pos_train,
        'X_student_lips': X_student_lips_train,
        #'X_teacher': X_teacher_train[:1],
        'y': y_train
    }
    val_data = {
        'X_student_hand_shape': X_student_hand_shape_val,
        'X_student_hand_pos': X_student_hand_pos_val,
        'X_student_lips': X_student_lips_val,
        #'X_teacher': X_teacher_val[:1],
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
    #X_teacher_tensors = data['X_teacher']
    y_tensors = data['y']
    
    # Create a TensorDataset with inputs and labels
    dataset = TensorDataset(
        X_student_hand_shape_tensors,
        X_student_hand_pos_tensors,
        X_student_lips_tensors,
        #X_teacher_tensors,
        y_tensors
    )
    
    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


class IPADataset(Dataset):
    def __init__(self, manifest_file, alphabet_file, sample_rate=16000, n_mels=80):
        self.manifest = pd.read_csv(manifest_file, header=None)
        with open(alphabet_file, "r") as f:
            self.alphabet = f.read().splitlines()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        # Step 1: Create phoneme_to_index with all phonemes (including space)
        self.phoneme_to_index = {phoneme: idx for idx, phoneme in enumerate(list(set(self.alphabet)))}
        self.phoneme_to_index[" "] = len(self.phoneme_to_index)  # Add space character

        # Step 2: Create index_to_phoneme after phoneme_to_index is fully populated
        self.index_to_phoneme = {idx: phoneme for phoneme, idx in self.phoneme_to_index.items()}

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
            ipa = f.read().strip().split()
        
        # Convert IPA to indices
        ipa_indices = [self.phoneme_to_index[phoneme] for phoneme in ipa]
        return torch.tensor(mel_spec, dtype=torch.float32), torch.tensor(ipa_indices, dtype=torch.long), 
    

class DeepSpeech2(nn.Module):
    def __init__(self, num_classes, hidden_size=1024, num_layers=5, n_mels=80):
        super(DeepSpeech2, self).__init__()
        
        # 2D Convolutional Layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        # Batch Normalization for Convolutional Layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Bidirectional LSTM Layers
        self.lstm = nn.LSTM(
            input_size=32 * n_mels,  # Output size of the last convolutional layer
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
        )
        
        # Batch Normalization for LSTM Output
        self.bn_lstm = nn.BatchNorm1d(hidden_size * 2)  # *2 for bidirectional LSTM
        
        # Fully Connected Layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional LSTM
        
    def forward(self, x):
        # Convolutional Layers with BatchNorm and ReLU
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Reshape for LSTM
        batch_size, channels, n_mels, time_steps = x.size()
        x = x.permute(0, 3, 1, 2)  # Move time_steps to the second dimension
        x = x.reshape(batch_size, time_steps, -1)  # Flatten the last two dimensions
        
        # LSTM Layers
        x, _ = self.lstm(x)
        
        # Apply BatchNorm to LSTM output
        x = x.permute(0, 2, 1)  # Swap dimensions for BatchNorm
        x = self.bn_lstm(x)
        x = x.permute(0, 2, 1)  # Swap back to original dimensions
        
        # Fully Connected Layer
        x = self.fc(x)
        
        return x
    
    

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import librosa


def collate_fn(batch):
    """
    Collate function to pad mel spectrograms and IPA indices to the same length.
    Args:
        batch: List of tuples (mel_spec, ipa_indices).
    Returns:
        Padded mel spectrograms and IPA indices.
    """
    # Separate mel spectrograms and IPA indices
    mel_specs, ipa_indices = zip(*batch)
    
    # Transpose mel spectrograms to have time dimension first
    mel_specs = [mel_spec.T for mel_spec in mel_specs]  # Transpose to [time, n_mels]
    
    # Pad mel spectrograms to the same length
    mel_specs_padded = pad_sequence(mel_specs, batch_first=True, padding_value=0)
    
    # Pad IPA indices to the same length
    ipa_indices_padded = pad_sequence(ipa_indices, batch_first=True, padding_value=teacher_dataset.phoneme_to_index[" "])

    
    return mel_specs_padded, ipa_indices_padded

# Student model
import torch
import torch.nn as nn

class StudentModel(nn.Module):
    def __init__(self, hand_shape_dim, hand_pos_dim, lips_dim, hidden_dim, output_dim):
        super(StudentModel, self).__init__()
        
        # Feature extractors for 1D feature vectors
        self.hand_shape_extractor = nn.Sequential(
            nn.Linear(hand_shape_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            #nn.Linear(128, 64),
            #nn.ReLU(),
            #nn.BatchNorm1d(64),
        )
        
        self.hand_pos_extractor = nn.Sequential(
            nn.Linear(hand_pos_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
        )
        
        self.lips_extractor = nn.Sequential(
            nn.Linear(lips_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            #nn.Linear(128, 64),
            #nn.ReLU(),
            #nn.BatchNorm1d(64),
        )
        
        # BiLSTM layers
        self.bilstm = nn.LSTM(
            input_size=64 * 2,  # Combined feature size (128 + 128)
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2,  # Add dropout for regularization
        )
        
        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),  # BiLSTM output is hidden_dim * 2 (bidirectional)
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim),
        )

    def forward(self, hand_shape, hand_pos, lips):
        # Get batch size and sequence length
        batch_size, seq_len, _ = hand_shape.shape
        
        # Reshape input tensors to (batch_size * seq_len, feature_dim)
        hand_shape = hand_shape.view(-1, hand_shape.size(-1))  # (batch_size * seq_len, hand_shape_dim)
        hand_pos = hand_pos.view(-1, hand_pos.size(-1))        # (batch_size * seq_len, hand_pos_dim)
        lips = lips.view(-1, lips.size(-1))                    # (batch_size * seq_len, lips_dim)
        
        # Extract features
        hand_shape_features = self.hand_shape_extractor(hand_shape)  # (batch_size * seq_len, 128)
        hand_pos_features = self.hand_pos_extractor(hand_pos)        # (batch_size * seq_len, 128)
        lips_features = self.lips_extractor(lips)                    # (batch_size * seq_len, 128)
        
        # Reshape features back to (batch_size, seq_len, feature_dim)
        hand_shape_features = hand_shape_features.view(batch_size, seq_len, -1)  # (batch_size, seq_len, 128)
        hand_pos_features = hand_pos_features.view(batch_size, seq_len, -1)      # (batch_size, seq_len, 128)
        lips_features = lips_features.view(batch_size, seq_len, -1)              # (batch_size, seq_len, 128)
        
        # Combine features
        combined_features = torch.cat([hand_shape_features, lips_features], dim=-1)  # (batch_size, seq_len, 256)
        
        # Pass through BiLSTM
        lstm_out, _ = self.bilstm(combined_features)  # (batch_size, seq_len, hidden_dim * 2)
        
        # Final predictions
        output = self.fc(lstm_out)  # (batch_size, seq_len, output_dim)
        return output

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
    #teacher_logits = teacher_logits.to(device)
    batch_y = batch_y.to(device)
    input_lengths = input_lengths.to(device)
    label_lengths = label_lengths.to(device)
    
    # Cosine similarity loss
    #cosine_loss = 1 - F.cosine_similarity(student_logits, teacher_logits, dim=-1).mean()
    
    # CTC loss
    log_probs = F.log_softmax(student_logits, dim=-1)  # Log-softmax of student_logits
    log_probs = log_probs.permute(1, 0, 2)  # Reshape to [sequence_length, batch_size, num_classes]
    
    ctc_loss = F.ctc_loss(
        log_probs,
        batch_y,  # Padded labels
        input_lengths,
        label_lengths,  # Lengths of label sequences (excluding padding)
        blank=phoneme_to_index[" "],  # Blank token index
    )
    
    # Combine losses
    total_loss = ctc_loss #+ cosine_loss
    #print(f"Cosine Loss: {cosine_loss.item()}, CTC Loss: {ctc_loss.item()}")
    return total_loss

import torch.nn.functional as F

def validate_student_model(student_model, val_loader, device):
    student_model.eval()  # Set student model to evaluation mode
    val_loss = 0.0

    with torch.no_grad():
        for batch_X_student_hand_shape, batch_X_student_hand_pos, batch_X_student_lips, batch_y in val_loader:
            # Move data to device
            batch_X_student_hand_shape = batch_X_student_hand_shape.to(device)
            batch_X_student_hand_pos = batch_X_student_hand_pos.to(device)
            batch_X_student_lips = batch_X_student_lips.to(device)
            batch_y = batch_y.to(device)

            # Forward pass through the student model
            student_logits = student_model(batch_X_student_hand_shape, batch_X_student_hand_pos, batch_X_student_lips)
            
            # Forward pass through the teacher model
            #with torch.no_grad():
            #    teacher_logits = teacher_model(batch_X_teacher.to(device).unsqueeze(1).permute(0, 1, 3, 2))  # Ensure teacher model outputs logits
            
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
            loss = sequence_level_distillation_loss(student_logits, None, batch_y, input_lengths, label_lengths, device)
            
            # Accumulate loss
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss

def train_student_model(student_model, teacher_model, train_loader, val_loader, num_epochs=50, device="cuda"):
      # Set teacher model to evaluation mode
    
    for epoch in range(num_epochs):
        student_model.train()
        #teacher_model.eval()
        epoch_loss = 0.0
        
        for batch_X_student_hand_shape, batch_X_student_hand_pos, batch_X_student_lips, batch_y in train_loader:
            # Move data to device
            batch_X_student_hand_shape = batch_X_student_hand_shape.to(device)
            batch_X_student_hand_pos = batch_X_student_hand_pos.to(device)
            batch_X_student_lips = batch_X_student_lips.to(device)
            #batch_X_teacher = batch_X_teacher.to(device)
            batch_y = batch_y.to(device)

            # Reshape batch_X_teacher for the teacher model
            #batch_X_teacher = batch_X_teacher.unsqueeze(1)  # Add channel dimension
            #batch_X_teacher = batch_X_teacher.permute(0, 1, 3, 2)  # Transpose to (batch_size, 1, num_mel_bins, time_steps)
            
            # Forward pass through the student model
            student_logits = student_model(batch_X_student_hand_shape, batch_X_student_hand_pos, batch_X_student_lips)
            
            # Forward pass through the teacher model
            #with torch.no_grad():
            #    teacher_logits = teacher_model(batch_X_teacher)  # Ensure teacher model outputs logits
            teacher_logits = None
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

        # Evaluate the model on the validation set every 10 epochs
        if (epoch + 1) % 5 == 0:
            val_loss = validate_student_model(student_model, val_loader, device)
            print(f"Validation Loss after Epoch [{epoch+1}/{num_epochs}]: {val_loss}")
    
    print("Training complete.")

# Example call to train_student_model function
# train_student_model(student_model, teacher_model, train_loader, val_loader, num_epochs=50)

def evaluate_student_model(student_model, val_loader):
    student_model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch_X_student_hand_shape, batch_X_student_hand_pos, batch_X_student_lips, batch_y in val_loader:
            # Move data to device
            batch_X_student_hand_shape = batch_X_student_hand_shape.to(device)
            batch_X_student_hand_pos = batch_X_student_hand_pos.to(device)
            batch_X_student_lips = batch_X_student_lips.to(device)
            #batch_X_teacher = batch_X_teacher.to(device)
            batch_y = batch_y.to(device)
            
            # Reshape batch_X_teacher for the teacher model
            #batch_X_teacher = batch_X_teacher.unsqueeze(1)  # Add channel dimension
            #batch_X_teacher = batch_X_teacher.permute(0, 1, 3, 2)  # Transpose to (batch_size, 1, num_mel_bins, time_steps)
            
            # Forward pass through the student model
            student_logits = student_model(batch_X_student_hand_shape, batch_X_student_hand_pos, batch_X_student_lips)
            
            # Forward pass through the teacher model
            #teacher_logits = teacher_model(batch_X_teacher)  # Ensure teacher model outputs logits
            
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
            loss = sequence_level_distillation_loss(student_logits, None, batch_y, input_lengths, label_lengths, device)
            total_loss += loss.item()
    
    print(f"Validation Loss: {total_loss / len(val_loader)}")

import torch.optim as optim
from torch.nn import CTCLoss
import torch.nn.functional as F

def validate_model(model, val_loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0

    with torch.no_grad():
        for val_mel_spec, val_ipa_indices in val_loader:
            # Move data to device
            val_mel_spec = val_mel_spec.to(device)
            val_ipa_indices = val_ipa_indices.to(device)

            # Transpose mel spectrograms back to [batch, n_mels, time]
            val_mel_spec = val_mel_spec.permute(0, 2, 1)

            # Forward pass
            val_outputs = model(val_mel_spec.unsqueeze(1))  # Add channel dimension
            val_outputs = val_outputs.permute(1, 0, 2)  # CTC expects (time, batch, num_classes)

            # Compute CTC loss
            val_input_lengths = torch.full((val_mel_spec.size(0),), val_outputs.size(0), dtype=torch.long, device=device)
            val_target_lengths = torch.tensor([len(ipa[ipa != teacher_dataset.phoneme_to_index[" "]]) for ipa in val_ipa_indices], dtype=torch.long, device=device)
            val_loss += criterion(F.log_softmax(val_outputs, dim=-1), val_ipa_indices, val_input_lengths, val_target_lengths).item()

    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss

def train_teacher_model(model, dataloader, val_loader, num_epochs=100, learning_rate=1e-3, device="cuda"):
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = CTCLoss()
    model.to(device)

    for epoch in range(num_epochs):  # Number of epochs
        model.train()  # Set model to training mode
        epoch_loss = 0.0  # Accumulate loss over the epoch

        for batch_idx, (mel_spec, ipa_indices) in enumerate(dataloader):
            # Move data to device
            mel_spec = mel_spec.to(device)
            ipa_indices = ipa_indices.to(device)

            # Transpose mel spectrograms back to [batch, n_mels, time]
            mel_spec = mel_spec.permute(0, 2, 1)

            # Forward pass
            outputs = model(mel_spec.unsqueeze(1))  # Add channel dimension
            outputs = outputs.permute(1, 0, 2)  # CTC expects (time, batch, num_classes)

            # Compute CTC loss
            input_lengths = torch.full((mel_spec.size(0),), outputs.size(0), dtype=torch.long, device=device)
            target_lengths = torch.tensor([len(ipa[ipa != teacher_dataset.phoneme_to_index[" "]]) for ipa in ipa_indices], dtype=torch.long, device=device)
            loss = criterion(F.log_softmax(outputs, dim=-1), ipa_indices, input_lengths, target_lengths)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()

        # Print average loss for the epoch
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss}")

        # Evaluate the model on the validation set every 10 epochs
        if (epoch + 1) % 5 == 0:
            val_loss = validate_model(model, val_loader, criterion, device)
            print(f"Validation Loss after Epoch [{epoch+1}/{num_epochs}]: {val_loss}")

    print("Training complete.")
        
    # Save the model
    torch.save(model.state_dict(), "/scratch2/bsow/Documents/ACSR/output/saved_models/deepspeech2_pretrained.pth")


def greedy_decoder(output, blank):
    """
    Decode model outputs using a greedy decoder.

    Args:
        output (torch.Tensor): Model outputs of shape (batch_size, sequence_length, num_classes).
        blank (int): Index of the blank token.

    Returns:
        list: List of decoded sequences.
    """
    arg_maxes = torch.argmax(output, dim=2)  # Get the most likely class for each time step
    decodes = []
    for args in arg_maxes:
        decode = []
        previous_idx = None
        for index in args:
            if index != blank and (previous_idx is None or index != previous_idx):
                decode.append(index.item())  # Append non-blank and non-repeated tokens
            previous_idx = index
        decodes.append(decode)
    return decodes


def decode_loader(model, loader, blank, index_to_phoneme):
    """
    Decode outputs for all batches in a DataLoader and return both decoded and true sequences.

    Args:
        model (torch.nn.Module): Trained model.
        loader (torch.utils.data.DataLoader): DataLoader containing input data and labels.
        blank (int): Index of the blank token.
        index_to_phoneme (dict): Mapping from indices to phonemes.

    Returns:
        tuple: (decoded_sequences, true_sequences), where:
            - decoded_sequences: List of decoded phoneme sequences.
            - true_sequences: List of true phoneme sequences.
    """
    model.eval()  # Set the model to evaluation mode
    all_decoded_sequences = []
    all_true_sequences = []

    with torch.no_grad():  # Disable gradient computation
        for batch_X_student_hand_shape, batch_X_student_hand_pos, batch_X_student_lips, batch_y in loader:
            # Move data to device
            batch_X_student_hand_shape = batch_X_student_hand_shape.to(device)
            batch_X_student_hand_pos = batch_X_student_hand_pos.to(device)
            batch_X_student_lips = batch_X_student_lips.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass through the student model
            outputs = model(batch_X_student_hand_shape, batch_X_student_hand_pos, batch_X_student_lips)
            
            decoded_phoneme_sequences = greedy_decoder(outputs, blank=blank)  # Decode outputs
            decoded_phonemes = [[index_to_phoneme[idx] for idx in sequence] for sequence in decoded_phoneme_sequences]  # Convert indices to phonemes
            all_decoded_sequences.extend(decoded_phonemes)  # Add to the list of decoded sequences

            # Convert true labels to phoneme sequences
            true_phoneme_sequences = [[index_to_phoneme[idx.item()] for idx in sequence if idx != blank and 
                                       index_to_phoneme[idx.item()] != " "] for sequence in batch_y]
            all_true_sequences.extend(true_phoneme_sequences)  # Add to the list of true sequences

    return all_decoded_sequences, all_true_sequences

def teacher_decode_loader(model, loader, blank, index_to_phoneme):
    """
    Decode outputs for all batches in a DataLoader and return both decoded and true sequences.

    Args:
        model (torch.nn.Module): Trained model.
        loader (torch.utils.data.DataLoader): DataLoader containing input data and labels.
        blank (int): Index of the blank token.
        index_to_phoneme (dict): Mapping from indices to phonemes.

    Returns:
        tuple: (decoded_sequences, true_sequences), where:
            - decoded_sequences: List of decoded phoneme sequences.
            - true_sequences: List of true phoneme sequences.
    """
    model.eval()  # Set the model to evaluation mode
    all_decoded_sequences = []
    all_true_sequences = []

    with torch.no_grad():  # Disable gradient computation
        for batch_X_student_hand_shape, batch_X_student_hand_pos, batch_X_student_lips, batch_X_teacher, batch_y in loader:
            # Move data to device
            batch_X_teacher = batch_X_teacher.to(device)
            # Reshape batch_X_teacher for the teacher model
            batch_X_teacher = batch_X_teacher.unsqueeze(1)  # Add channel dimension
            batch_X_teacher = batch_X_teacher.permute(0, 1, 3, 2)  # Transpose to (batch_size, 1, num_mel_bins, time_steps)
            
            # Forward pass through the student model
            outputs = model(batch_X_teacher)
            
            decoded_phoneme_sequences = greedy_decoder(outputs, blank=blank)  # Decode outputs
            decoded_phonemes = [[index_to_phoneme[idx] for idx in sequence] for sequence in decoded_phoneme_sequences]  # Convert indices to phonemes
            all_decoded_sequences.extend(decoded_phonemes)  # Add to the list of decoded sequences

            # Convert true labels to phoneme sequences
            true_phoneme_sequences = [[index_to_phoneme[idx.item()] for idx in sequence if idx != blank and 
                                       index_to_phoneme[idx.item()] != " "] for sequence in batch_y]
            all_true_sequences.extend(true_phoneme_sequences)  # Add to the list of true sequences

    return all_decoded_sequences, all_true_sequences

import jiwer

def calculate_per_with_jiwer(decoded_sequences, true_sequences):
    """
    Calculate the Phoneme Error Rate (PER) using jiwer.

    Args:
        decoded_sequences (list): List of decoded phoneme sequences.
        true_sequences (list): List of true phoneme sequences.

    Returns:
        float: Phoneme Error Rate (PER).
    """
    # Convert phoneme sequences to space-separated strings
    decoded_str = [" ".join(seq) for seq in decoded_sequences]
    true_str = [" ".join(seq) for seq in true_sequences]

    # Calculate PER using jiwer
    per = jiwer.wer(true_str, decoded_str)
    return per


if __name__ == "__main__":
    # Directories
    data_dir = r'/scratch2/bsow/Documents/ACSR/output/predictions'
    phoneme_dir = r'/scratch2/bsow/Documents/ACSR/data/training_videos/CSF22_train/train_labels'
    audio_dir = r'/scratch2/bsow/Documents/ACSR/data/training_videos/audio'
    textgrid_dir = r'/scratch2/bsow/Documents/ACSR/data/training_videos/textgrids'
    video_dir = r'/scratch2/bsow/Documents/ACSR/data/training_videos/videos'
    coordinates_dir = r'/scratch2/bsow/Documents/ACSR/output/extracted_coordinates'
    labels_dir = r'/scratch2/bsow/Documents/ACSR/data/training_videos/CSF22_train/train_labels'

    coordinates_data = load_csv_files(coordinates_dir, "_coordinates")
    # Find phoneme files
    base_names = coordinates_data.keys()
    phoneme_files = find_phoneme_files(phoneme_dir, base_names)
    print("Number of phoneme files found:", len(phoneme_files))

    # Prepare data
    all_videos_data = prepare_data_for_videos_no_sliding_windows(
        base_names, phoneme_files, audio_dir, textgrid_dir, video_dir, coordinates_dir, labels_dir, phoneme_to_index
    )

    # Combine sequences with padding
    X_student_hand_shape_padded, X_student_hand_pos_padded, X_student_lips_padded, y_padded = combine_sequences_with_padding(all_videos_data)

    # Convert to PyTorch tensors
    X_student_hand_shape_tensor = torch.tensor(X_student_hand_shape_padded, dtype=torch.float32)
    X_student_hand_pos_tensor = torch.tensor(X_student_hand_pos_padded, dtype=torch.float32)
    X_student_lips_tensor = torch.tensor(X_student_lips_padded, dtype=torch.float32)
    #X_teacher_tensor = torch.tensor(X_teacher_padded, dtype=torch.float32)
    y_tensor = torch.tensor(y_padded, dtype=torch.long)

    # Final organized data
    all_videos_data = {
        "X_student_hand_shape": X_student_hand_shape_tensor,  # Hand shape coordinates
        "X_student_hand_pos": X_student_hand_pos_tensor,      # Hand position coordinates
        "X_student_lips": X_student_lips_tensor,              # Lip landmarks
        #"X_teacher": X_teacher_tensor,                        # Audio features (log-mel spectrogram)
        "y": y_tensor,                                        # Phoneme labels
    }

    # Split data
    train_data, val_data = train_val_split(all_videos_data)

    # Prepare DataLoaders
    train_loader = data_to_dataloader(train_data, batch_size=16, shuffle=True)
    val_loader = data_to_dataloader(val_data, batch_size=8, shuffle=False)

    print("Len of train dataset", len(train_data['X_student_hand_shape']))
    print("Len of val dataset", len(val_data['X_student_hand_shape']))

    # Check the DataLoader output
    for batch_X_student_hand_shape, batch_X_student_hand_pos, batch_X_student_lips, batch_y in train_loader:
        print("Batch X_student_hand_shape shape:", batch_X_student_hand_shape.shape)
        print("Batch X_student_hand_pos shape:", batch_X_student_hand_pos.shape)
        print("Batch X_student_lips shape:", batch_X_student_lips.shape)
        #print("Batch X_teacher shape:", batch_X_teacher.shape)
        print("Batch y shape:", batch_y.shape)
        break

    # Initialize the student model
    student_model = StudentModel(
        hand_shape_dim=63,  # 21 keypoints × 3 coordinates
        hand_pos_dim=63,     # 3 coordinates (x, y, z)
        lips_dim=117,       # 40 keypoints × 3 coordinates
        hidden_dim=256,
        output_dim=len(phoneme_to_index)  # Number of phonemes
    )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #teacher_dataset = IPADataset(manifest_file="/scratch2/bsow/Documents/ACSR/data/train.csv", alphabet_file="/scratch2/bsow/Documents/ACSR/data/training_videos/phoneme_dictionary.txt")
    #teacher_dataloader = DataLoader(teacher_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    ## Load the pretrained teacher model
    #teacher_model = DeepSpeech2(num_classes=len(teacher_dataset.phoneme_to_index))
    #teacher_model.load_state_dict(torch.load("/scratch2/bsow/Documents/ACSR/output/saved_models/deepspeech2_pretrained.pth", map_location=torch.device('cpu')))
    #teacher_model.to(device)
    #train_teacher_model(teacher_model, teacher_dataloader, num_epochs=200, device=device)

    # teacher model output for one batch
    #for mel_spec, ipa_indices in teacher_dataloader:
    #    mel_spec = mel_spec.to(device)
    #    mel_spec = mel_spec.permute(0, 2, 1)
    #    outputs = teacher_model(mel_spec.unsqueeze(1))
    #    print("Teacher model output shape:", outputs.shape)
    #    decoded_phoneme_sequences = greedy_decoder(outputs, blank=teacher_dataset.phoneme_to_index[" "])
    #    decoded_phonemes = [[teacher_dataset.index_to_phoneme[idx] for idx in sequence] for sequence in decoded_phoneme_sequences]
    #    print("Teacher model decoded phoneme sequences:", decoded_phonemes[:6])
    #    true_sentences = [[teacher_dataset.index_to_phoneme[idx.item()] for idx in sequence if idx.item() != teacher_dataset.phoneme_to_index[" "]] for sequence in ipa_indices]
    #    print("True phoneme sequences:", true_sentences[:6])
    #    break

    # Set teacher model to evaluation mode
    #teacher_model.eval()
    print("Training on device:", device)
    student_model.to(device)
    #teacher_model.to(device)

    # Optimizer
    optimizer = Adam(student_model.parameters(), lr=1e-3)
    # Start training
    train_student_model(student_model, None, train_loader, val_loader, num_epochs=110, device=device)

    evaluate_student_model(student_model, val_loader)

    blank_token =  phoneme_to_index[" "]
    decoded_train_sequences, true_train_sequences = decode_loader(student_model, train_loader, blank_token, index_to_phoneme)
    decoded_val_sequences, true_val_sequences = decode_loader(student_model, val_loader, blank_token, index_to_phoneme)

    # Print results
    print("Decoded training phoneme sequences:", decoded_train_sequences[:5])
    print("True training phoneme sequences:", true_train_sequences[:5])
    print("Decoded validation phoneme sequences:", decoded_val_sequences[:5])
    print("True validation phoneme sequences:", true_val_sequences[:5])

    train_per = calculate_per_with_jiwer(decoded_train_sequences, true_train_sequences)
    val_per = calculate_per_with_jiwer(decoded_val_sequences, true_val_sequences)

    #decoded_train_sequences, true_train_sequences = teacher_decode_loader(teacher_model, train_loader, blank_token, index_to_phoneme)
    #decoded_val_sequences, true_val_sequences = teacher_decode_loader(teacher_model, val_loader, blank_token, index_to_phoneme)
    #print("Teacher model results:")
    #print('Decoded training phoneme sequences:', decoded_train_sequences[:5])
    #print('True training phoneme sequences:', true_train_sequences[:5])
    #print('Decoded validation phoneme sequences:', decoded_val_sequences[:5])
    #print('True validation phoneme sequences:', true_val_sequences[:5])

    print("Training PER (jiwer):", train_per, "1 - PER: ", 1 - train_per)
    print("Validation PER (jiwer):", val_per, "1 - PER: ", 1 - val_per)

    # Save the trained student model
    torch.save(student_model.state_dict(), "/scratch2/bsow/Documents/ACSR/output/saved_models/student_model.pth")
    print("Student model saved.")