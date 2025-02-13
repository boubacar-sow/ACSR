import csv
import os
from collections import defaultdict

import jiwer
import numpy as np
import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from praatio import textgrid as tgio
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

import wandb  # Import W&B
from next_syllable_prediction import NextSyllableLSTM
from variables import consonant_to_handshape, vowel_to_position

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

# Find corresponding phoneme files based on the base names of position filenames
def find_phoneme_files(directory, base_names):
    phoneme_files = {}
    for base_name in base_names:
        phoneme_file = os.path.join(directory, f'{base_name}.csv')
        if os.path.exists(phoneme_file):
            phoneme_files[base_name] = phoneme_file
    return phoneme_files

def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]

# Load phoneme-to-index mapping
# with open(r"/scratch2/bsow/Documents/ACSR/data/training_videos/CSF22_train/phonelist.csv", "r") as file:
with open(r"/scratch2/bsow/Documents/ACSR/data/news_dataset/vocab.txt", "r") as file:
    reader = csv.reader(file)
    vocabulary_list = [row[0] for row in reader]

phoneme_to_index = {phoneme: idx for idx, phoneme in enumerate(unique(vocabulary_list))}
index_to_phoneme = {idx: phoneme for phoneme, idx in phoneme_to_index.items()}

def load_features(directory, base_name):
    file_path = os.path.join(directory, f"{base_name}_features.csv")
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)  # Drop rows with NaN values
    return df

def syllabify_ipa(ipa_text):
    consonants = {"b", "d", "f", "g", "h", "j", "k", "l", "m", "n", "n~", "p", "r", "s", "s^", "t", "v", "w", "z", "z^", "ng", "gn"}
    vowels = {"a", "a~", "e", "e^", "e~", "i", "o", "o^", "o~", "u", "y", "x", "x^", "x~"}
    phonemes = ipa_text.split()
    syllables = []
    i = 0

    while i < len(phonemes):
        phone = phonemes[i]
        if phone in vowels:
            syllables.append(phone)
            i += 1
        elif phone in consonants:
            # Check if there is a next phone
            if i + 1 < len(phonemes):
                next_phone = phonemes[i + 1]
                if next_phone in vowels:
                    syllable = phone + next_phone
                    syllables.append(syllable)
                    i += 2 
                else:
                    syllables.append(phone)
                    i += 1
            else:
                syllables.append(phone)
                i += 1
        else:
            i += 1
    return syllables


from collections import Counter


def prepare_data_for_videos_no_sliding_windows(base_names, phoneme_files, features_dir, labels_dir, 
                                              phoneme_to_index, img_size=(28, 28),
                                              lips_dir="/scratch2/bsow/Documents/ACSR/data/training_videos/CSF22_train/lip_rois_mp4"):
    all_videos_data = {}
    syllable_counter = Counter()
    
    for base_name in base_names:
        if base_name in phoneme_files:
            # Load pre-extracted features
            features_df = load_features(features_dir, base_name)
            if 'frame_number' not in features_df.columns:
                raise ValueError(f"Feature file for {base_name} does not contain 'frame_number' column.")

            # Get valid frame numbers after NaN removal
            valid_frames = features_df['frame_number'].astype(int).tolist()
            
            # Load and process lip images
            video_lips_dir = os.path.join(lips_dir, base_name, base_name)
            X_visual_lips = []
            
            #for frame_num in valid_frames:
            #    img_path = os.path.join(video_lips_dir, f"{base_name}_lips_{frame_num-1:04d}.png")
            #    if not os.path.exists(img_path):
            #        raise FileNotFoundError(f"Lip image not found: {img_path}")
            #    
            #    # Load and preprocess image
            #    img = cv2.imread(img_path)
            #    img = cv2.resize(img, img_size)
            #    img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
            #    X_visual_lips.append(img)
            #
            #X_visual_lips = np.array(X_visual_lips)

            # Separate features into different modalities
            hand_shape_columns = [col for col in features_df.columns if "hand" in col and "face" not in col]
            hand_pos_columns = [col for col in features_df.columns if "face" in col]
            lip_columns = [col for col in features_df.columns if "lip" in col]

            # Extract features
            X_acoustic_hand_shape = features_df[hand_shape_columns].to_numpy()
            X_acoustic_hand_pos = features_df[hand_pos_columns].to_numpy()
            X_acoustic_lips = features_df[lip_columns].to_numpy()

            # Load and process phoneme labels
            labels_path = os.path.join(labels_dir, f"{base_name}.csv")
            if not os.path.exists(labels_path):
                raise FileNotFoundError(f"Phoneme label file not found: {labels_path}")
            
            phoneme_labels = pd.read_csv(labels_path, header=None).squeeze().tolist()[1:-1]
            syllable_labels = ["<SOS>"] + syllabify_ipa(" ".join(phoneme_labels)) + ["<EOS>"]
            
            # Convert to indices
            syllable_indices = []
            for syllable in syllable_labels:
                if syllable not in phoneme_to_index:
                    raise ValueError(f"Syllable '{syllable}' not found in vocabulary. File: {base_name}")
                syllable_indices.append(phoneme_to_index[syllable])
                syllable_counter[syllable] += 1

            # Store all data modalities
            all_videos_data[base_name] = {
                "X_acoustic_hand_shape": X_acoustic_hand_shape,
                "X_acoustic_hand_pos": X_acoustic_hand_pos,
                "X_acoustic_lips": X_acoustic_lips,
                "X_visual_lips": None,  # New visual modality
                "y": syllable_indices,
            }
    
    return all_videos_data, syllable_counter

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
    # Get the number of samples
    num_samples = len(data['X_acoustic_hand_shape'])
    split_idx = int(num_samples * train_ratio)
    
    # If the indices are already saved, load them
    if os.path.exists("/scratch2/bsow/Documents/ACSR/src/acsr/indices.npy"):
        print("Loading indices from file")
        indices = np.load("/scratch2/bsow/Documents/ACSR/src/acsr/indices.npy")
    else:
        # Randomize the data
        indices = np.random.permutation(num_samples)
        np.save("/scratch2/bsow/Documents/ACSR/src/acsr/indices.npy", indices)
    
    # Helper function to split data based on indices
    def split_data(key):
        return [data[key][i] for i in indices]
    
    # Split hand shape features
    X_acoustic_hand_shape = split_data('X_acoustic_hand_shape')
    X_acoustic_hand_shape_train = X_acoustic_hand_shape[:split_idx]
    X_acoustic_hand_shape_val = X_acoustic_hand_shape[split_idx:]
    
    # Split hand position features
    X_acoustic_hand_pos = split_data('X_acoustic_hand_pos')
    X_acoustic_hand_pos_train = X_acoustic_hand_pos[:split_idx]
    X_acoustic_hand_pos_val = X_acoustic_hand_pos[split_idx:]
    
    # Split acoustic lip features
    X_acoustic_lips = split_data('X_acoustic_lips')
    X_acoustic_lips_train = X_acoustic_lips[:split_idx]
    X_acoustic_lips_val = X_acoustic_lips[split_idx:]
    
    # Split visual lip features
    #X_visual_lips = split_data('X_visual_lips')
    #X_visual_lips_train = X_visual_lips[:split_idx]
    #X_visual_lips_val = X_visual_lips[split_idx:]
    
    # Split labels
    y = split_data('y')
    y_train = y[:split_idx]
    y_val = y[split_idx:]
    
    # Create train and validation data dictionaries
    train_data = {
        'X_acoustic_hand_shape': X_acoustic_hand_shape_train,
        'X_acoustic_hand_pos': X_acoustic_hand_pos_train,
        'X_acoustic_lips': X_acoustic_lips_train,
        'X_visual_lips': [],  # Include visual lips
        'y': y_train
    }
    val_data = {
        'X_acoustic_hand_shape': X_acoustic_hand_shape_val,
        'X_acoustic_hand_pos': X_acoustic_hand_pos_val,
        'X_acoustic_lips': X_acoustic_lips_val,
        'X_visual_lips': [],  # Include visual lips
        'y': y_val
    }
    return train_data, val_data

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset


import torch
from torch.nn.utils.rnn import pad_sequence

def custom_collate_fn(batch):
    hand_shape, hand_pos, acoustic_lips, labels = zip(*batch)
    
    # Pad sequences for hand shape, hand position, and acoustic lips
    hand_shape_padded = pad_sequence([torch.tensor(x, dtype=torch.float32) for x in hand_shape], batch_first=True, padding_value=0)
    hand_pos_padded = pad_sequence([torch.tensor(x, dtype=torch.float32) for x in hand_pos], batch_first=True, padding_value=0)
    acoustic_lips_padded = pad_sequence([torch.tensor(x, dtype=torch.float32) for x in acoustic_lips], batch_first=True, padding_value=0)
    
    # Pad visual lips (image sequences)
    #visual_lips_padded = pad_sequence(
    #    [torch.tensor(x, dtype=torch.float32) for x in visual_lips],
    #    batch_first=True,
    #    padding_value=0  # Use zero-padding for images
    #)
    
    # Pad labels
    labels_padded = pad_sequence([torch.tensor(y, dtype=torch.long) for y in labels], batch_first=True, padding_value=phoneme_to_index["<UNK>"])
    
    return hand_shape_padded, hand_pos_padded, acoustic_lips_padded, labels_padded

from torch.utils.data import DataLoader

def data_to_dataloader(data, batch_size=4, shuffle=True):
    dataset = list(zip(
        data['X_acoustic_hand_shape'],
        data['X_acoustic_hand_pos'],
        data['X_acoustic_lips'],
        #data['X_visual_lips'],  # Include visual lips
        data['y']
    ))
    
    # Create a DataLoader with the custom collate function
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=custom_collate_fn  # Use the custom collate function
    )
    return dataloader


# acoustic model
import torch.nn as nn
from conformer import ConformerBlock

class ThreeStreamFusionEncoder(nn.Module):
    def __init__(self, hand_shape_dim, hand_pos_dim, lips_dim, visual_lips_dim, hidden_dim=128, n_layers=2):
        super(ThreeStreamFusionEncoder, self).__init__()

        self.hand_shape_gru = nn.GRU(hand_shape_dim, hidden_dim, n_layers, bidirectional=True, batch_first=True)
        self.hand_pos_gru   = nn.GRU(hand_pos_dim, hidden_dim, n_layers, bidirectional=True, batch_first=True)
        self.lips_gru       = nn.GRU(lips_dim, hidden_dim, n_layers, bidirectional=True, batch_first=True)
        
        # Fusion GRU: note the input size is 3 streams * 2 (bidirectional)
        self.fusion_gru = nn.GRU(hidden_dim * 6, hidden_dim * 3, n_layers, bidirectional=True, batch_first=True)
        
        # CNN for visual lips (unchanged)
        self.visual_lips_cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, hidden_dim)),  # (batch, 64, 1, 64)
            nn.Flatten(start_dim=1),                # (batch, 4096)
            nn.Linear(128 * hidden_dim, hidden_dim)   # nn.Linear(4096, 64) if hidden_dim=64
        )

        # Cross-modal attention and fusion conformer remain the same
        self.cross_modal_attention = nn.MultiheadAttention(embed_dim=hidden_dim*6, num_heads=4)
        #self.fusion_conformer = nn.Sequential(*[ConformerBlock(dim=hidden_dim * 3) for _ in range(n_layers)])
    
    def forward(self, hand_shape, hand_pos, lips, visual_lips):
        # Project and process each modality with its respective Conformer
        hand_shape_out, _ = self.hand_shape_gru(hand_shape)  # (batch, seq, hidden_dim*2)
        hand_pos_out, _   = self.hand_pos_gru(hand_pos)
        lips_out, _       = self.lips_gru(lips)
   
        # visual_lips has shape (batch, seq_len, height, width, channels)
        #batch_size, seq_len, H, W, C = visual_lips.shape
        #visual_lips = visual_lips.permute(0, 1, 4, 2, 3)
        #visual_lips = visual_lips.reshape(-1, C, H, W)
        #visual_lips_out = self.visual_lips_cnn(visual_lips)
        #visual_lips_out = visual_lips_out.reshape(batch_size, seq_len, -1)
        
        # Concatenate all modalities along the last dimension
        combined_features = torch.cat([hand_shape_out, hand_pos_out, lips_out], dim=-1)
        
        # Apply cross-modal attention
        #attn_output, _ = self.cross_modal_attention(combined_features, combined_features, combined_features)
        
        # Fusion Conformer
        fusion_out, _ = self.fusion_gru(combined_features)
        return fusion_out


class AttentionDecoder(nn.Module):
    def __init__(self, encoder_dim, output_dim, hidden_dim_decoder=None, n_layers=1):
        super(AttentionDecoder, self).__init__()
        # If not provided, set hidden_dim_decoder equal to encoder_dim
        if hidden_dim_decoder is None:
            hidden_dim_decoder = encoder_dim
        self.embedding = nn.Embedding(output_dim, hidden_dim_decoder)
        self.gru = nn.GRU(hidden_dim_decoder + encoder_dim, hidden_dim_decoder, n_layers, batch_first=True)
        self.out = nn.Linear(hidden_dim_decoder, output_dim)
        self.encoder_dim = encoder_dim
        self.hidden_dim_decoder = hidden_dim_decoder

        # If needed, add a projection if hidden_dim_decoder != encoder_dim.
        # self.proj = nn.Linear(hidden_dim_decoder, encoder_dim)

    def forward(self, encoder_outputs, target_seq):
        """
        Args:
            encoder_outputs: Tensor of shape (batch, T, encoder_dim)
            target_seq: Tensor of shape (batch, target_len) containing target indices
        Returns:
            outputs: Tensor of shape (batch, target_len, output_dim)
        """
        batch_size, target_len = target_seq.size()
        hidden = None  # Alternatively, initialize hidden state here.
        outputs = []

        # For each time step in the target sequence (using teacher forcing)
        for t in range(target_len):
            # Get embedding for current target token: shape (batch, 1, hidden_dim_decoder)
            embedded = self.embedding(target_seq[:, t].long()).unsqueeze(1)
            
            # Dot-product attention:
            # Compute attention scores by dot-product between embedded and all encoder outputs.
            # embedded: (batch, 1, hidden_dim_decoder)
            # If hidden_dim_decoder != encoder_dim, you might project embedded via self.proj first.
            attn_scores = torch.bmm(embedded, encoder_outputs.transpose(1, 2))  # shape: (batch, 1, T)
            attn_weights = F.softmax(attn_scores, dim=-1)  # shape: (batch, 1, T)
            
            # Compute context vector as weighted sum of encoder outputs: shape (batch, 1, encoder_dim)
            attn_applied = torch.bmm(attn_weights, encoder_outputs)
            
            # Concatenate embedded input and context vector
            gru_input = torch.cat([embedded, attn_applied], dim=2)  # shape: (batch, 1, hidden_dim_decoder + encoder_dim)
            
            # Pass through GRU
            output, hidden = self.gru(gru_input, hidden)  # output: (batch, 1, hidden_dim_decoder)
            output = self.out(output.squeeze(1))  # shape: (batch, output_dim)
            outputs.append(output)
        
        outputs = torch.stack(outputs, dim=1)  # (batch, target_len, output_dim)
        return outputs


class JointCTCAttentionModel(nn.Module):
    def __init__(self, hand_shape_dim, hand_pos_dim, lips_dim, visual_lips_dim, output_dim, hidden_dim=128):
        super(JointCTCAttentionModel, self).__init__()
        self.encoder = ThreeStreamFusionEncoder(hand_shape_dim, hand_pos_dim, lips_dim, visual_lips_dim, hidden_dim)
        self.attention_decoder = AttentionDecoder(encoder_dim=hidden_dim * 6, output_dim=output_dim)
        self.ctc_fc = nn.Linear(hidden_dim * 6, output_dim)
    
    def forward(self, hand_shape, hand_pos, lips, visual_lips, target_seq=None):
        encoder_out = self.encoder(hand_shape, hand_pos, lips, visual_lips)
        ctc_logits = self.ctc_fc(encoder_out)
        if target_seq is not None:
            attn_logits = self.attention_decoder(encoder_out, target_seq)
        else:
            attn_logits = None
        return ctc_logits, attn_logits

import sys

def joint_ctc_attention_loss(ctc_logits, attn_logits, target_seq, input_lengths, label_lengths, alpha, device):
    # CTC loss branch
    log_probs = F.log_softmax(ctc_logits, dim=-1).permute(1, 0, 2)
    ctc_loss = F.ctc_loss(
        log_probs,
        target_seq,       # Padded labels (expected as LongTensor)
        input_lengths,
        label_lengths,
        blank=phoneme_to_index["<UNK>"],
    )
    
    # Attention branch loss
    attn_loss = F.cross_entropy(
        attn_logits.view(-1, attn_logits.size(-1)),
        target_seq.view(-1),
        ignore_index=phoneme_to_index["<UNK>"]
    )
    
    total_loss = alpha * ctc_loss + (1 - alpha) * attn_loss
    return total_loss, ctc_loss, attn_loss


def validate_model(model, val_loader, alpha, device):
    model.eval()
    val_loss = 0.0
    total_ctc_loss = 0.0
    total_attn_loss = 0.0

    with torch.no_grad():
        for batch_X_hand_shape, batch_X_hand_pos, batch_X_lips, batch_y in val_loader:
            # Ensure batch_y is of type long
            batch_X_hand_shape = batch_X_hand_shape.to(device)
            batch_X_hand_pos = batch_X_hand_pos.to(device)
            batch_X_lips = batch_X_lips.to(device)
            #batch_X_visual_lips = batch_X_visual_lips.to(device)  # Visual lips
            batch_y = batch_y.long().to(device)
            
            # Forward pass with teacher forcing
            ctc_logits, attn_logits = model(
                batch_X_hand_shape, batch_X_hand_pos, batch_X_lips, None, target_seq=batch_y
            )
            # Compute input_lengths and label_lengths
            input_lengths = torch.full(
                (batch_X_hand_shape.size(0),),
                ctc_logits.size(1),
                dtype=torch.long,
                device=device
            )
            label_lengths = (batch_y != phoneme_to_index["<UNK>"]).sum(dim=1).to(device)
            
            # Compute combined loss
            loss, ctc_loss, attn_loss = joint_ctc_attention_loss(
                ctc_logits, attn_logits, batch_y, input_lengths, label_lengths, alpha, device
            )
            val_loss += loss.item()
            total_ctc_loss += ctc_loss.item()
            total_attn_loss += attn_loss.item()

    avg_val_loss = val_loss / len(val_loader)
    avg_ctc_loss = total_ctc_loss / len(val_loader)
    avg_attn_loss = total_attn_loss / len(val_loader)
    return avg_val_loss, avg_ctc_loss, avg_attn_loss

import time 

def train_model(model, train_loader, val_loader, optimizer, num_epochs, alpha, device):
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        epoch_loss = 0.0
        epoch_ctc_loss = 0.0
        epoch_attn_loss = 0.0
        
        for batch_X_hand_shape, batch_X_hand_pos, batch_X_lips, batch_y in train_loader:
            # Convert targets to LongTensor and move data to device
            batch_X_hand_shape = batch_X_hand_shape.to(device)
            batch_X_hand_pos = batch_X_hand_pos.to(device)
            batch_X_lips = batch_X_lips.to(device)
            #batch_X_visual_lips = batch_X_visual_lips.to(device)  # Visual lips
            batch_y = batch_y.long().to(device)
            
            # Forward pass with teacher forcing
            ctc_logits, attn_logits = model(
                batch_X_hand_shape, batch_X_hand_pos, batch_X_lips, None, target_seq=batch_y
            )
            
            # Compute input_lengths and label_lengths
            input_lengths = torch.full(
                (batch_X_hand_shape.size(0),),
                ctc_logits.size(1),
                dtype=torch.long,
                device=device
            )
            label_lengths = (batch_y != phoneme_to_index["<UNK>"]).sum(dim=1).to(device)
            
            # Compute joint loss
            loss, ctc_loss, attn_loss = joint_ctc_attention_loss(
                ctc_logits, attn_logits, batch_y, input_lengths, label_lengths, alpha, device
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_ctc_loss += ctc_loss.item()
            epoch_attn_loss += attn_loss.item()
        
        # Logging to W&B (or print)
        train_loss_avg = epoch_loss / len(train_loader)
        wandb.log({"train_loss": train_loss_avg, "epoch": epoch + 1})
        wandb.log({"train_ctc_loss": epoch_ctc_loss / len(train_loader), "epoch": epoch + 1})
        wandb.log({"train_attn_loss": epoch_attn_loss / len(train_loader), "epoch": epoch + 1})
        
        # Validate the model
        val_loss, val_ctc_loss, val_attn_loss = validate_model(model, val_loader, alpha, device)
        wandb.log({"val_loss": val_loss, "epoch": epoch + 1})
        wandb.log({"val_ctc_loss": val_ctc_loss, "epoch": epoch + 1})
        wandb.log({"val_attn_loss": val_attn_loss, "epoch": epoch + 1})
        
        # Optionally, decode and compute PER or WER on the validation set
        blank_token = phoneme_to_index["<UNK>"]
        decoded_val_sequences, true_val_sequences, true_val_gestures, decoded_val_gestures = decode_loader(model, val_loader, blank_token, index_to_phoneme, device)
        val_per = calculate_per_with_jiwer(decoded_val_sequences, true_val_sequences)
        val_gestures_per = calculate_per_with_jiwer(decoded_val_gestures, true_val_gestures)
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {round(train_loss_avg, 3)}, "
              f"Val Loss: {round(val_loss, 3)}, Accuracy (1-PER): {round(1 - val_per, 3)}, Accuracy gestures (1-PER): {round(1 - val_gestures_per, 3)}, "
              f"Time: {round(time.time() - epoch_start_time, 2)} sec")
        sys.stdout.flush()
        wandb.log({"val_per": 1 - val_per, "epoch": epoch + 1})
        wandb.log({"val_wer": val_per, "epoch": epoch + 1})
        
        if epoch % 10 == 0 and epoch > 3000:
            torch.save(model.state_dict(), f"/scratch2/bsow/Documents/ACSR/src/acsr/model_epoch3.pt")
    
    print("Training complete.")

def syllables_to_gestures(syllable_sequence):
    """
    Convert a sequence of syllables into a sequence of gestures.
    
    Args:
        syllable_sequence (list): A list of syllables (strings).
        
    Returns:
        list: A list of gesture strings in the format "handshape-position".
    """
    gestures = []
    for syllable in syllable_sequence:
        if syllable == "<SOS>" or syllable == "<EOS>" or syllable == "<PAD>" or syllable == "<UNK>":
            gestures.append(syllable)
        # Check if the syllable starts with a multi-character consonant (e.g., "s^")
        elif len(syllable) >= 3 and syllable[:2] in consonant_to_handshape:
            consonant = syllable[:2]
            vowel = syllable[2:]  # Remaining part is the vowel
            handshape = consonant_to_handshape.get(consonant, 5)  # Default handshape is 5
            position = vowel_to_position.get(vowel, 1)  # Default position is 1
            gestures.append(f"{handshape}-{position}")
        # Check if the syllable ends with a multi-character vowel (e.g., "me^")
        elif len(syllable) >= 3 and syllable[-2:] in vowel_to_position:
            consonant = syllable[:-2]  # Remaining part is the consonant
            vowel = syllable[-2:]
            handshape = consonant_to_handshape.get(consonant, 5)  # Default handshape is 5
            position = vowel_to_position.get(vowel, 1)  # Default position is 1
            gestures.append(f"{handshape}-{position}")
        # Handle normal CV syllables (e.g., "ma")
        elif len(syllable) == 2:
            if syllable in consonant_to_handshape:  # length 2 consonant only syllable
                handshape = consonant_to_handshape.get(syllable, 5)  # Default handshape is 5
                position = 1  # Default position is 1
                gestures.append(f"{handshape}-{position}")
            elif syllable in vowel_to_position:  # length 2 vowel only syllable
                handshape = 5  # Default handshape is 5
                position = vowel_to_position.get(syllable, 1)
                gestures.append(f"{handshape}-{position}")
            elif syllable[0] in consonant_to_handshape:  # Consonant-Vowel pair
                consonant = syllable[0]
                vowel = syllable[1]
                handshape = consonant_to_handshape.get(consonant, 5)  # Default handshape is 5
                position = vowel_to_position.get(vowel, 1)  # Default position is 1
                gestures.append(f"{handshape}-{position}")
            elif syllable[0] in vowel_to_position:  # Vowel-only syllable
                vowel = syllable
                position = vowel_to_position.get(vowel, 1)  # Default position is 1
                gestures.append(f"5-{position}")  # Default handshape is 5
        # Handle C-only syllables (e.g., "m")
        elif len(syllable) == 1 and syllable in consonant_to_handshape:
            handshape = consonant_to_handshape.get(syllable, 5)  # Default handshape is 5
            gestures.append(f"{handshape}-1")  # Default position is 1
        # Handle V-only syllables (e.g., "a")
        elif len(syllable) == 1 and syllable in vowel_to_position:
            position = vowel_to_position.get(syllable, 1)  # Default position is 1
            gestures.append(f"5-{position}")  # Default handshape is 5
        else:
            # Unknown syllable
            print(f"Unknown syllable: {syllable}")
    return gestures
import torch
import torch.nn.functional as F

def remove_duplicates_and_blanks(seq, blank_idx):
    """
    Remove consecutive duplicate tokens and then remove all blank tokens.
    """
    cleaned_seq = []
    prev_token = None
    for token in seq:
        if token == blank_idx:
            prev_token = None
        elif token != prev_token:
            cleaned_seq.append(token)
            prev_token = token
    return cleaned_seq

def remove_blanks(seq, blank_idx):
    """Remove blank tokens from a candidate sequence."""
    return [s for s in seq if s != blank_idx]

def compute_lm_score(seq, nextsyllable_model, sos_idx, pad_idx, max_seq_len, device):
    if not seq:
        return 0.0
    lm_score = 0.0
    # Prepend the <SOS> token
    #seq = [sos_idx] + seq  
    for i in range(1, len(seq)):
        prefix = seq[:i]
        if len(prefix) < max_seq_len:
            padded_prefix = [pad_idx] * (max_seq_len - len(prefix)) + prefix
        else:
            padded_prefix = prefix[-max_seq_len:]
        input_tensor = torch.tensor(padded_prefix, dtype=torch.long, device=device).unsqueeze(0)
        lm_logits = nextsyllable_model(input_tensor)  # shape: (1, vocab_size)
        lm_log_probs = F.log_softmax(lm_logits, dim=-1)
        token_log_prob = lm_log_probs[0, seq[i]].item()
        lm_score += token_log_prob
    return lm_score

import torch

def logsumexp(a, b):
    # Combine two log values in a numerically stable manner using torch.logaddexp.
    return torch.logaddexp(torch.tensor(a), torch.tensor(b)).item()

def remove_duplicates_and_blanks(seq, blank_idx):
    """
    Remove consecutive duplicate tokens and then remove blanks.
    """
    cleaned_seq = []
    prev_token = None
    for token in seq:
        if token == blank_idx:
            prev_token = None  # Reset on blank
        elif token != prev_token:
            cleaned_seq.append(token)
            prev_token = token
    return cleaned_seq

def ctc_beam_search(ctc_logits, beam_width, blank_idx):
    """
    A simplified beam search over the CTC outputs.
    
    Args:
        ctc_logits: Tensor of shape (T, V) for one sample,
                    where T is the number of time steps and V is the vocabulary size.
        beam_width: Number of candidates to keep at each time step.
        blank_idx: Index of the blank token.
        
    Returns:
        A list of tuples (candidate_seq, acoustic_score) where candidate_seq is a list of token indices.
        The acoustic_score is the (log) probability of that candidate.
    """
    T, V = ctc_logits.shape
    beams = {(): 0.0}  # Start with an empty sequence and score zero (log prob = 0)

    for t in range(T):
        new_beams = {}
        log_probs_t = torch.log_softmax(ctc_logits[t], dim=0)  # shape (V,)

        for seq, score in beams.items():
            topk_log_probs, topk_indices = torch.topk(log_probs_t, beam_width)
            for i in range(len(topk_indices)):
                token = topk_indices[i].item()
                token_log_prob = topk_log_probs[i].item()
                new_score = score + token_log_prob

                new_seq = seq + (token,)  # Always add the token, including blank

                if new_seq in new_beams:
                    new_beams[new_seq] = max(new_beams[new_seq], new_score)
                else:
                    new_beams[new_seq] = new_score

        # Keep only top beam_width candidates
        beams = dict(sorted(new_beams.items(), key=lambda x: x[1], reverse=True)[:1500])

    # Remove consecutive duplicate tokens between blanks
    final_candidates = [(remove_duplicates_and_blanks(seq, blank_idx), score) for seq, score in beams.items()]

    return final_candidates

def beam_search_decode(cuedspeech_model, nextsyllable_model, inputs_hand_shape, inputs_hand_pos, inputs_lips, inputs_visual_lips,
                       blank_idx, index_to_syllable, beam_width=5, alpha=0.7, device="cuda", max_seq_len=15, test=False):
    """
    Run beam search decoding on a single sample.
    """
    cuedspeech_model.eval()
    with torch.no_grad():
        ctc_logits, _ = cuedspeech_model(
            inputs_hand_shape.to(device),
            inputs_hand_pos.to(device),
            inputs_lips.to(device),
            inputs_visual_lips.to(device)
        )
    # Process only the first sample (batch=1)
    ctc_logits = ctc_logits[0]
    pad_idx = phoneme_to_index["<PAD>"]
    sos_idx = phoneme_to_index["<SOS>"]
    candidates = ctc_beam_search(ctc_logits, beam_width=beam_width, blank_idx=blank_idx)

    # Stage 2: Rescore candidates with the LM.
    rescored_candidates = []
    for seq, acoustic_score in candidates:
        clean_seq = remove_blanks(seq, blank_idx)
        lm_score = compute_lm_score(clean_seq, nextsyllable_model, sos_idx, pad_idx, max_seq_len, device)
        combined_score = acoustic_score + alpha * lm_score
        rescored_candidates.append((clean_seq, combined_score))
    
    if test:
        for seq, score in rescored_candidates:
            syllables = [index_to_syllable[idx] for idx in seq]
            print(" ".join(syllables), "       ----         Score:", score)
    # save the rescored candidates in a file
    with open("/scratch2/bsow/Documents/ACSR/src/acsr/rescored_candidates.txt", "w") as file:
        for seq, score in rescored_candidates:
            syllables = [index_to_syllable[idx] for idx in seq]
            file.write(" ".join(syllables) + "       ----         Score: " + str(score) + "\n")

    best_candidate, best_score = max(rescored_candidates, key=lambda x: x[1])
    decoded_sequence = [index_to_syllable[idx] for idx in best_candidate]
    return decoded_sequence

def decode_loader_beam(cuedspeech_model, nextsyllable_model, loader, blank, index_to_syllable, device="cuda",
                       beam_width=5, alpha=0.4, max_seq_len=15, test=False):
    """
    Decode all samples from a DataLoader using beam search decoding.
    """
    cuedspeech_model.eval()
    all_decoded_sequences = []
    all_true_sequences = []
    with torch.no_grad():
        for batch_X_hand_shape, batch_X_hand_pos, batch_X_lips, batch_y in loader:
            batch_X_hand_shape = batch_X_hand_shape.to(device)
            batch_X_hand_pos = batch_X_hand_pos.to(device)
            batch_X_lips = batch_X_lips.to(device)
            batch_X_visual_lips = batch_X_visual_lips.to(device)  # Visual lips
            batch_y = batch_y.long().to(device)
            
            for i in range(batch_X_hand_shape.size(0)):
                sample_hand_shape = batch_X_hand_shape[i].unsqueeze(0)
                sample_hand_pos   = batch_X_hand_pos[i].unsqueeze(0)
                sample_lips       = batch_X_lips[i].unsqueeze(0)
                sample_visual_lips = batch_X_visual_lips[i].unsqueeze(0)  # Visual lips
                decoded_sequence = beam_search_decode(cuedspeech_model, nextsyllable_model,
                                                      sample_hand_shape, sample_hand_pos, sample_lips, sample_visual_lips,
                                                      blank_idx=blank, index_to_syllable=index_to_syllable,
                                                      beam_width=beam_width, alpha=alpha, device=device,
                                                      max_seq_len=max_seq_len, test=test)
                all_decoded_sequences.append(decoded_sequence)
            for sequence in batch_y:
                seq_syllables = [index_to_syllable[idx.item()] for idx in sequence if idx != blank and index_to_syllable[idx.item()] != " "]
                all_true_sequences.append(seq_syllables)
    return all_decoded_sequences, all_true_sequences

def greedy_decoder(output, blank, index_to_phoneme):
    probs = F.softmax(output, dim=-1)
    arg_maxes = torch.argmax(probs, dim=2)
    top10_probs, top10_indices = torch.topk(probs, k=10, dim=-1)
    
    raw_decodes = []
    collapsed_decodes = []
    
    batch_size, time_steps = arg_maxes.shape
    for batch_idx in range(batch_size):
        raw_seq = []
        collapsed_seq = []
        current_candidate = None  # (token, max_prob, raw_info)
        
        for t in range(time_steps):
            current_index = arg_maxes[batch_idx, t].item()
            token_prob = probs[batch_idx, t, current_index].item()
            current_token = index_to_phoneme.get(current_index, "<UNK>")
            
            # Build raw info
            raw_info = {
                "token": current_token,
                "prob": token_prob,
                "top2_token": index_to_phoneme.get(top10_indices[batch_idx, t, 1].item(), "<UNK>"),
                "top2_prob": top10_probs[batch_idx, t, 1].item(),
                "top3_token": index_to_phoneme.get(top10_indices[batch_idx, t, 2].item(), "<UNK>"),
                "top3_prob": top10_probs[batch_idx, t, 2].item(),
                "top4_token": index_to_phoneme.get(top10_indices[batch_idx, t, 3].item(), "<UNK>"),
                "top4_prob": top10_probs[batch_idx, t, 3].item(),
                "top5_token": index_to_phoneme.get(top10_indices[batch_idx, t, 4].item(), "<UNK>"),
                "top5_prob": top10_probs[batch_idx, t, 4].item(),
                "top6_token": index_to_phoneme.get(top10_indices[batch_idx, t, 5].item(), "<UNK>"),
                "top6_prob": top10_probs[batch_idx, t, 5].item(),
                "top7_token": index_to_phoneme.get(top10_indices[batch_idx, t, 6].item(), "<UNK>"),
                "top7_prob": top10_probs[batch_idx, t, 6].item(),
                "top8_token": index_to_phoneme.get(top10_indices[batch_idx, t, 7].item(), "<UNK>"),
                "top8_prob": top10_probs[batch_idx, t, 7].item(),
                "timestep": t
            }
            if raw_info["token"] == "<UNK>" and raw_info["prob"] < 0.98: # if the model is not confident about the prediction of the blank token
                raw_info["token"], raw_info["top2_token"] = raw_info["top2_token"], raw_info["token"]
                raw_info["prob"], raw_info["top2_prob"] = raw_info["top2_prob"], raw_info["prob"]
            raw_seq.append(raw_info)
            
            # Collapse logic
            if current_index == blank:
                if current_candidate:
                    collapsed_seq.append(current_candidate["raw_info"])
                    current_candidate = None
                continue
                
            if current_candidate:
                if current_token == current_candidate["token"]:
                    # Keep the highest probability occurrence
                    if token_prob > current_candidate["max_prob"]:
                        current_candidate = {
                            "token": current_token,
                            "max_prob": token_prob,
                            "raw_info": raw_info
                        }
                else:
                    # Finalize previous candidate
                    collapsed_seq.append(current_candidate["raw_info"])
                    current_candidate = {
                        "token": current_token,
                        "max_prob": token_prob,
                        "raw_info": raw_info
                    }
            else:
                current_candidate = {
                    "token": current_token,
                    "max_prob": token_prob,
                    "raw_info": raw_info
                }
        
        # Add final candidate if exists
        if current_candidate:
            collapsed_seq.append(current_candidate["raw_info"])
        
        raw_decodes.append(raw_seq)
        collapsed_decodes.append(collapsed_seq)
    
    return raw_decodes, collapsed_decodes

def decode_loader(model, loader, blank, index_to_phoneme, device='cuda', training=False):
    """
    Decode sequences from a DataLoader using the CTC branch of the joint model.
    This function now obtains two outputs from greedy_decoder: the raw per-timestep
    top-5 tokens (including blanks) and the collapsed (final) predictions.
    
    Args:
        model: The joint model.
        loader: DataLoader yielding batches.
        blank (int): The blank token index.
        index_to_phoneme (dict): Mapping from indices to phonemes.
        device (str): e.g., 'cuda' or 'cpu'.
        training (bool): If False, prints the decoded outputs.
    
    Returns:
        tuple: (all_collapsed_tokens, all_true_sequences, all_decoded_gestures, all_true_gestures)
               where all_collapsed_tokens is the list of collapsed decoded sequences.
    """
    model.eval()
    all_raw_decoded_sequences = []       # To store raw top-5 info per timestep
    all_collapsed_decoded_sequences = [] # To store final collapsed sequences
    all_true_sequences = []
    all_decoded_gestures = []
    all_true_gestures = []
    
    with torch.no_grad():
        for batch_X_hand_shape, batch_X_hand_pos, batch_X_lips, batch_y in loader:
            batch_X_hand_shape = batch_X_hand_shape.to(device)
            batch_X_hand_pos = batch_X_hand_pos.to(device)
            batch_X_lips = batch_X_lips.to(device)
            batch_y = batch_y.to(device)
            
            ctc_logits, _ = model(batch_X_hand_shape, batch_X_hand_pos, batch_X_lips, None)
            raw_decodes, collapsed_decodes = greedy_decoder(ctc_logits, blank=blank, 
                                                              index_to_phoneme=index_to_phoneme)
            
            # Use the collapsed output for further processing.
            decoded_phonemes = [[token_info["token"] for token_info in seq] 
                                for seq in collapsed_decodes]
            decoded_gestures = [syllables_to_gestures(seq) for seq in decoded_phonemes]
            
            all_raw_decoded_sequences.extend(raw_decodes)
            all_collapsed_decoded_sequences.extend(collapsed_decodes)
            all_decoded_gestures.extend(decoded_gestures)
            
            # Process true labels
            true_phoneme_sequences = []
            for sequence in batch_y:
                seq_phonemes = [
                    index_to_phoneme[idx.item()]
                    for idx in sequence 
                    if idx != blank and index_to_phoneme[idx.item()] != " "
                ]
                true_phoneme_sequences.append(seq_phonemes)
            all_true_sequences.extend(true_phoneme_sequences)
            all_true_gestures.extend([syllables_to_gestures(seq) for seq in true_phoneme_sequences])
    
    # If not in training mode, print out the raw (per-timestep) top-5 information for a few samples.
    if not training:
        output_file = "/scratch2/bsow/Documents/ACSR/src/acsr/rescored_candidates.txt"
        # Open the file in write mode (this will overwrite it) or use "a" for appending.
        with open(output_file, "w") as f:
            print("Raw decoded sequences (per timestep top-5 tokens, including blanks):", file=f)
            for i, raw_seq in enumerate(all_raw_decoded_sequences[:15]):
                print(f"Sample {i}:", file=f)
                for t, token_info in enumerate(raw_seq):
                    print(f"  Timestep {t}: {token_info}", file=f)

        print("\nCollapsed decoded syllable sequences:")
        for i, collapsed_seq in enumerate(all_collapsed_decoded_sequences[-5:]):
            print(f"Sample {i}: {[token_info['token'] for token_info in collapsed_seq]}")
        
        print("\nDecoded validation gesture sequences:")
        for i, gesture_seq in enumerate(all_decoded_gestures[-5:]):
            print(f"Sample {i}: {gesture_seq}")
    
    # For evaluation, we return only the collapsed tokens.
    all_collapsed_tokens = [[token_info["token"] for token_info in seq] 
                            for seq in all_collapsed_decoded_sequences]
    
    return all_collapsed_tokens, all_true_sequences, all_decoded_gestures, all_true_gestures

import itertools

import itertools
import torch
import torch.nn.functional as F
import sys
import gc

def rescore_sequences(collapsed_decodes, raw_decodes, true_sequences, phoneme_to_index, nextsyllable_model, device, threshold=0.8, top_k=6, batch_size=10000):
    sos_idx = phoneme_to_index["<SOS>"]
    pad_idx = phoneme_to_index["<PAD>"]
    max_seq_len = 15
    nextsyllable_model.eval()
    all_rescored_sequences = []
    
    for sample_idx, (collapsed_seq, raw_seq, true_seq) in enumerate(zip(collapsed_decodes, raw_decodes, true_sequences)):
        alternatives = []
        for token_info in collapsed_seq:
            if token_info['prob'] < threshold:
                t = token_info['timestep']
                raw_token_info = raw_seq[t]
                topk_tokens = [raw_token_info["token"]]
                for i in range(2, top_k + 1):
                    topk_token = raw_token_info.get(f'top{i}_token', "<UNK>")
                    topk_tokens.append(topk_token)
                alternatives.append(topk_tokens[:top_k])
            else:
                alternatives.append([token_info['token']])
        
        candidate_sequences = list(itertools.product(*alternatives))
        # remove <UNK> tokens in the candidate for each candidate sequence
        candidate_sequences = [[token for token in candidate if token != "<UNK>"] for candidate in candidate_sequences]
        print(f"Sample {sample_idx}, {len(candidate_sequences)} candidate sequences.")
        sys.stdout.flush()
        if not candidate_sequences:
            best_sequence = [token_info['token'] for token_info in collapsed_seq]
            all_rescored_sequences.append(best_sequence)
            continue

        # Convert candidates to indices and track original lengths
        candidate_indices_list = []
        original_lengths = []  # Track actual sequence lengths
        for candidate in candidate_sequences:
            indices = []
            for token in candidate:
                if token in phoneme_to_index:
                    indices.append(phoneme_to_index[token])
                else:
                    indices.append(phoneme_to_index.get("<UNK>", pad_idx))
            original_lengths.append(len(indices))  # Record original length
            candidate_indices_list.append(indices)

        # Pad sequences to maximum length in this sample
        max_L = max(original_lengths)
        padded_candidates = []
        for seq in candidate_indices_list:
            if len(seq) < max_L:
                padded_seq = seq + [pad_idx] * (max_L - len(seq))
            else:
                padded_seq = seq
            padded_candidates.append(padded_seq)

        # Process in batches
        lm_scores = torch.zeros(len(candidate_sequences), device=device)

        for batch_start in range(0, len(candidate_sequences), batch_size):
            batch_end = min(batch_start + batch_size, len(candidate_sequences))
            batch_candidates = torch.tensor(padded_candidates[batch_start:batch_end], device=device, dtype=torch.long)
            batch_lengths = torch.tensor(original_lengths[batch_start:batch_end], device=device)

            for i in range(1, max_L):
                # Mask to ignore sequences where i >= actual length
                valid_mask = (i < batch_lengths).float()  # 1.0 if valid, else 0.0

                # Extract prefixes (includes padding but will be masked)
                prefixes = batch_candidates[:, :i]

                # Pad/truncate to max_seq_len for model input
                if i < max_seq_len:
                    pad = torch.full((batch_candidates.size(0), max_seq_len - i), pad_idx, device=device, dtype=torch.long)
                    padded_prefixes = torch.cat([pad, prefixes], dim=1)
                elif i > max_seq_len:
                    padded_prefixes = prefixes[:, (i - max_seq_len):i]
                else:
                    padded_prefixes = prefixes

                # Get model predictions
                with torch.no_grad():
                    lm_logits = nextsyllable_model(padded_prefixes)
                torch.cuda.empty_cache()
                gc.collect()
                lm_log_probs = F.log_softmax(lm_logits, dim=-1)

                # Get targets and compute log probs
                targets = batch_candidates[:, i]
                selected_log_probs = lm_log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

                # Apply mask to ignore invalid positions (i >= sequence length)
                selected_log_probs *= valid_mask

                # Accumulate scores
                lm_scores[batch_start:batch_end] += selected_log_probs

        # Find best sequence
        best_idx = torch.argmax(lm_scores).item()
        best_sequence = candidate_sequences[best_idx]
        all_rescored_sequences.append(list(best_sequence))
        print("Best sequence: ", best_sequence)
        print("Original sequ: ", [token_info["token"] for token_info in collapsed_seq])
        print("True sequence: ", true_seq)
        print()
        sys.stdout.flush()
    
    return all_rescored_sequences

# decode_loader_with_rescoring remains unchanged

def decode_loader_with_rescoring(model, nextsyllable_model, loader, blank, index_to_phoneme, phoneme_to_index, device='cuda'):
    model.eval()
    raw_decodes_all = []
    collapsed_decodes_all = []
    true_sequences = []
    
    with torch.no_grad():
        for batch_X_hand_shape, batch_X_hand_pos, batch_X_lips, batch_y in loader:
            # ... [same as original decode_loader] ...
            batch_X_hand_shape = batch_X_hand_shape.to(device)
            batch_X_hand_pos = batch_X_hand_pos.to(device)
            batch_X_lips = batch_X_lips.to(device)
            batch_y = batch_y.to(device)
            ctc_logits, _ = model(batch_X_hand_shape, batch_X_hand_pos, batch_X_lips, None)
            raw_decodes, collapsed_decodes = greedy_decoder(ctc_logits, blank, index_to_phoneme)
            
            raw_decodes_all.extend(raw_decodes)
            collapsed_decodes_all.extend(collapsed_decodes)
            # Process true labels
            true_phoneme_sequences = []
            for sequence in batch_y:
                seq_phonemes = [
                    index_to_phoneme[idx.item()]
                    for idx in sequence 
                    if idx != blank and index_to_phoneme[idx.item()] != " "
                ]
                true_phoneme_sequences.append(seq_phonemes)
            true_sequences.extend(true_phoneme_sequences)
    
    # Rescore sequences
    rescored_sequences = rescore_sequences(
        collapsed_decodes_all, raw_decodes_all, true_sequences,
        phoneme_to_index, nextsyllable_model, device
    )
    
    return rescored_sequences, true_sequences

def calculate_per_with_jiwer(decoded_sequences, true_sequences):
    # Convert phoneme sequences to space-separated strings
    decoded_str = [" ".join(seq) for seq in decoded_sequences]
    true_str = [" ".join(seq) for seq in true_sequences]
    # Calculate PER using jiwer
    try:
        per = jiwer.wer(true_str, decoded_str)
    except Exception as e:
        print("Error calculating PER:", e)
        print("True sequences:", true_str)
        print("Decoded sequences:", decoded_str)
    return per

if __name__ == "__main__":
    # Directories
    data_dir = r'/scratch2/bsow/Documents/ACSR/output/predictions'
    phoneme_dir = r'/scratch2/bsow/Documents/ACSR/data/training_videos/CSF22_train/train_labels'
    coordinates_dir = r'/scratch2/bsow/Documents/ACSR/output/extracted_coordinates'
    features_dir = r'/scratch2/bsow/Documents/ACSR/output/extracted_features'
    labels_dir = r'/scratch2/bsow/Documents/ACSR/data/training_videos/CSF22_train/train_labels'

    features_data = load_csv_files(features_dir, "_features")
    # Find phoneme files
    base_names = features_data.keys()
    phoneme_files = find_phoneme_files(phoneme_dir, base_names)
    print("Number of phoneme files found:", len(phoneme_files))

    # Prepare data
    all_videos_data, syllable_counter = prepare_data_for_videos_no_sliding_windows(
        base_names, phoneme_files, features_dir, labels_dir, phoneme_to_index
    )

    syllable_df = pd.DataFrame.from_dict(syllable_counter, orient='index', columns=['frequency'])
    syllable_df.index.name = 'syllable'
    syllable_df.reset_index(inplace=True)
    syllable_df = syllable_df.sort_values(by='frequency', ascending=False)

    # Save the syllable distribution to a CSV file
    output_csv_path = os.path.join("/scratch2/bsow/Documents/ACSR/src/acsr", 'syllable_distribution.csv')
    syllable_df.to_csv(output_csv_path, index=False)
    print(f"Syllable distribution saved to {output_csv_path}")


    # Final organized data
    data = {
        "X_acoustic_hand_shape": [all_videos_data[video]["X_acoustic_hand_shape"] for video in all_videos_data],  # Hand shape coordinates
        "X_acoustic_hand_pos": [all_videos_data[video]["X_acoustic_hand_pos"] for video in all_videos_data],      # Hand position coordinates
        "X_acoustic_lips": [all_videos_data[video]["X_acoustic_lips"] for video in all_videos_data],              # Lip coordinates
        "X_visual_lips": [all_videos_data[video]["X_visual_lips"] for video in all_videos_data],                  # Visual lips
        "y": [all_videos_data[video]["y"] for video in all_videos_data],                                        # Phoneme labels
    }
    # Split data
    train_data, val_data = train_val_split(data)

    # Prepare DataLoaders
    train_loader = data_to_dataloader(train_data, batch_size=16, shuffle=True)
    val_loader = data_to_dataloader(val_data, batch_size=16, shuffle=False)

    print("Len of train dataset", len(train_data['X_acoustic_hand_shape']))
    print("Len of val dataset", len(val_data['X_acoustic_hand_shape']))

    # Check the DataLoader output
    for batch_X_acoustic_hand_shape, batch_X_acoustic_hand_pos, batch_X_acoustic_lips, batch_y in train_loader:
        print("Batch X_acoustic_hand_shape shape:", batch_X_acoustic_hand_shape.shape)
        print("Batch X_acoustic_hand_pos shape:", batch_X_acoustic_hand_pos.shape)
        print("Batch X_acoustic_lips shape:", batch_X_acoustic_lips.shape)
        #print("Batch X_visual_lips shape:", batch_X_visual_lips.shape)
        #print("Batch X_teacher shape:", batch_X_teacher.shape)
        print("Batch y shape:", batch_y.shape)
        print("Output dim of the model: ", len(phoneme_to_index))
        break

    learning_rate = 1e-3
    batch_size = 64
    hidden_dim_fusion = 256
    epochs = 5000
    encoder_hidden_dim = 128
    output_dim = len(phoneme_to_index)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    level = "syllables"
    n_layers_gru = 2
    alpha = 0.2
    wandb.login(key="580ab03d7111ed25410f9831b06b544b5f5178a2")
    # Initialize W&B
    #wandb.init(project="acsr", config={
    #    "learning_rate": learning_rate,
    #    "batch_size": batch_size,
    #    "epochs": epochs,
    #    "encoder_hidden_dim": encoder_hidden_dim,
    #    "output_dim": output_dim,
    #    "n_layers_gru": n_layers_gru,
    #    "alpha": alpha,
    #    "device": device,
    #    "level": level,
    #})

    # Define the model
    acoustic_model = JointCTCAttentionModel(
        hand_shape_dim=batch_X_acoustic_hand_shape.shape[-1],  # Number of hand shape keypoints
        hand_pos_dim=batch_X_acoustic_hand_pos.shape[-1],      # Number of hand position keypoints
        lips_dim=batch_X_acoustic_lips.shape[-1],              # Number of lip keypoints
        visual_lips_dim=None,        # Number of visual lip keypoints
        output_dim=output_dim,  # Number of phonemes
        hidden_dim=encoder_hidden_dim,  # Hidden dimension for GRUs
    )

    # Optimizer
    optimizer = Adam(acoustic_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Training on device:", device)
    acoustic_model.to(device)
    # load the trained acoustic model
    acoustic_model.load_state_dict(torch.load("/scratch2/bsow/Documents/ACSR/src/acsr/best_model.pt", map_location=device))
    acoustic_model.to(device)
    
    # Start training
    #train_model(acoustic_model, train_loader, val_loader, num_epochs=epochs, alpha=alpha, device=device, optimizer=optimizer)
    #torch.save(acoustic_model.state_dict(), "/scratch2/bsow/Documents/ACSR/output/saved_models/model_epoch3.pth")


    blank_token =  phoneme_to_index["<UNK>"]
    decoded_train_sequences, true_train_sequences, true_train_gestures, decoded_train_gestures = decode_loader(acoustic_model, train_loader, blank_token, index_to_phoneme, device, training=True)
    decoded_val_sequences, true_val_sequences, decoded_val_gestures, true_val_gestures = decode_loader(acoustic_model, val_loader, blank_token, index_to_phoneme, device, training=False)

    # Print results
    #print("Decoded training phoneme sequences:", decoded_train_sequences[:5])
    #print("True training phoneme sequences:", true_train_sequences[:5])
    print("Decoded validation phoneme sequences:", decoded_val_sequences[:5])
    print("True validation phoneme sequences:", true_val_sequences[:5])
    sys.stdout.flush()
    train_per = calculate_per_with_jiwer(decoded_train_sequences, true_train_sequences)
    val_per = calculate_per_with_jiwer(decoded_val_sequences, true_val_sequences)
    print("Training PER (jiwer):", train_per, "1 - PER: ", 1 - train_per)
    print("Validation PER (jiwer):", val_per, "1 - PER: ", 1 - val_per)

    #print("Decoded training gesture sequences:", decoded_train_gestures[:5])
    #print("True training gesture sequences:", true_train_gestures[:5])
    print("Decoded validation gesture sequences:", decoded_val_gestures[-5:])
    print("True validation gesture sequences:", true_val_gestures[-5:])

    # Calculate PER for gestures
    train_per_gestures = calculate_per_with_jiwer(decoded_train_gestures, true_train_gestures)
    val_per_gestures = calculate_per_with_jiwer(decoded_val_gestures, true_val_gestures)
    print("Training PER (jiwer) for gestures:", train_per_gestures, "1 - PER: ", 1 - train_per_gestures)
    print("Validation PER (jiwer) for gestures:", val_per_gestures, "1 - PER: ", 1 - val_per_gestures)
    sys.stdout.flush()
    
    print("="*210)

    # Initialize model
    nextsyllable_model = NextSyllableLSTM(
        vocab_size=len(phoneme_to_index),
        embedding_dim=200,
        hidden_dim=512,
        num_layers=4,
        dropout=0.2
    ).to(device)
####
    # Load model weights
    nextsyllable_model.load_state_dict(torch.load("/scratch2/bsow/Documents/ACSR/src/acsr/wandb/run-20250131_113223-rge6w8nh/files/best_syllable_model_def2.pth", map_location=device))
    # Ensure both models are on the same device
    nextsyllable_model.to(device)
    nextsyllable_model.eval()
    #
    # After training your models, perform decoding
    blank_token = phoneme_to_index["<UNK>"]
    beam_width = 15
    alpha = 0.5  # Adjust alpha to balance between models
    decoded_val_sequences

    # Rescored decoding
    rescored_sequences, _ = decode_loader_with_rescoring(acoustic_model, nextsyllable_model, val_loader, blank_token, index_to_phoneme, phoneme_to_index, device)

    # Calculate PER
    greedy_per = calculate_per_with_jiwer(decoded_val_sequences, true_val_sequences)
    rescored_per = calculate_per_with_jiwer(rescored_sequences, true_val_sequences)

    print(f"Greedy 1 - PER: {1 - greedy_per:.3f}, Rescored 1 - PER: {1 - rescored_per:.3f}")
    #Finish W&B run
    wandb.finish()

    