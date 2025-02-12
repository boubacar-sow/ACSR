import csv
import os
from collections import defaultdict

import librosa
import numpy as np
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


def prepare_data_for_videos_no_sliding_windows(base_names, phoneme_files, features_dir, labels_dir, phoneme_to_index):
    all_videos_data = {}
    syllable_counter = Counter()
    
    for base_name in base_names:
        if base_name in phoneme_files:
            # Load pre-extracted features
            features_df = load_features(features_dir, base_name)
            if 'frame_number' not in features_df.columns:
                raise ValueError(f"Feature file for {base_name} does not contain 'frame_number' column.")

            # Separate features into hand shape, hand position, and lip features
            hand_shape_columns = [col for col in features_df.columns if "hand" in col and "face" not in col]
            hand_pos_columns = [col for col in features_df.columns if "face" in col]
            lip_columns = [col for col in features_df.columns if "lip" in col]

            # Extract features
            X_acoustic_hand_shape = features_df[hand_shape_columns].to_numpy()
            X_acoustic_hand_pos = features_df[hand_pos_columns].to_numpy()
            X_acoustic_lips = features_df[lip_columns].to_numpy()

            # Load phoneme labels from CSV file
            labels_path = os.path.join(labels_dir, f"{base_name}.csv")
            if not os.path.exists(labels_path):
                raise FileNotFoundError(f"Phoneme label file not found: {labels_path}")
            
            # Read the CSV file
            phoneme_labels = pd.read_csv(labels_path, header=None).squeeze().tolist()  # Convert to list of phonemes
            # remove <start> and <end> tokens
            phoneme_labels = phoneme_labels[1:-1]
            # Convert phoneme labels to syllables
            syllable_labels = syllabify_ipa(" ".join(phoneme_labels))
            # add <SOS> and <EOS> tokens
            syllable_labels = ["<SOS>"] + syllable_labels + ["<EOS>"]

            # Convert syllable labels to indices
            syllable_indices = []
            for syllable in syllable_labels:
                if syllable not in phoneme_to_index:
                    raise ValueError(f"Syllable '{syllable}' not found in the vocabulary. File: {base_name}")
                syllable_indices.append(phoneme_to_index[syllable])
                syllable_counter[syllable] += 1  # Count syllable occurrence

            # Combine features and syllable indices
            all_videos_data[base_name] = {
                "X_acoustic_hand_shape": X_acoustic_hand_shape,  # Hand shape features
                "X_acoustic_hand_pos": X_acoustic_hand_pos,      # Hand position features
                "X_acoustic_lips": X_acoustic_lips,              # Lip features
                "y": syllable_indices,                         # Syllable labels (sequence)
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
    
    # if the indices are already saved, load them
    if os.path.exists("/scratch2/bsow/Documents/ACSR/src/acsr/indices.npy"):
        print("Loading indices from file")
        indices = np.load("/scratch2/bsow/Documents/ACSR/src/acsr/indices.npy")
    else:
        # Randomize the data
        indices = np.random.permutation(num_samples)
        np.save("/scratch2/bsow/Documents/ACSR/src/acsr/indices.npy", indices)

    # Split hand shape features
    X_acoustic_hand_shape = [data['X_acoustic_hand_shape'][i] for i in indices]
    X_acoustic_hand_shape_train = X_acoustic_hand_shape[:split_idx]
    X_acoustic_hand_shape_val = X_acoustic_hand_shape[split_idx:]

    # Split hand position features
    X_acoustic_hand_pos = [data['X_acoustic_hand_pos'][i] for i in indices]
    X_acoustic_hand_pos_train = X_acoustic_hand_pos[:split_idx]
    X_acoustic_hand_pos_val = X_acoustic_hand_pos[split_idx:]

    # Split lip features
    X_acoustic_lips = [data['X_acoustic_lips'][i] for i in indices]
    X_acoustic_lips_train = X_acoustic_lips[:split_idx]
    X_acoustic_lips_val = X_acoustic_lips[split_idx:]

    # Split labels
    y = [data['y'][i] for i in indices]
    y_train = y[:split_idx]
    y_val = y[split_idx:]

    # Create train and validation data dictionaries
    train_data = {
        'X_acoustic_hand_shape': X_acoustic_hand_shape_train,
        'X_acoustic_hand_pos': X_acoustic_hand_pos_train,
        'X_acoustic_lips': X_acoustic_lips_train,
        'y': y_train
    }
    val_data = {
        'X_acoustic_hand_shape': X_acoustic_hand_shape_val,
        'X_acoustic_hand_pos': X_acoustic_hand_pos_val,
        'X_acoustic_lips': X_acoustic_lips_val,
        'y': y_val
    }

    return train_data, val_data

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset


def custom_collate_fn(batch):
    # Unpack the batch
    hand_shape, hand_pos, lips, labels = zip(*batch)

    # Pad sequences to the maximum length in the batch
    hand_shape_padded = pad_sequence([torch.tensor(x, dtype=torch.float32) for x in hand_shape], batch_first=True, padding_value=0)
    hand_pos_padded = pad_sequence([torch.tensor(x, dtype=torch.float32) for x in hand_pos], batch_first=True, padding_value=0)
    lips_padded = pad_sequence([torch.tensor(x, dtype=torch.float32) for x in lips], batch_first=True, padding_value=0)

    # Pad labels with the padding token (e.g., phoneme_to_index["<UNK>"])
    labels_padded = pad_sequence([torch.tensor(y, dtype=torch.float32) for y in labels], batch_first=True, padding_value=phoneme_to_index["<UNK>"])

    return hand_shape_padded, hand_pos_padded, lips_padded, labels_padded

def data_to_dataloader(data, batch_size=4, shuffle=True):
    # Create a dataset from the lists
    dataset = list(zip(data['X_acoustic_hand_shape'], data['X_acoustic_hand_pos'], data['X_acoustic_lips'], data['y']))

    # Create a DataLoader with the custom collate function
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=custom_collate_fn  # Use the custom collate function
    )
    return dataloader


# acoustic model
class ThreeStreamFusionEncoder(nn.Module):
    def __init__(self, hand_shape_dim, hand_pos_dim, lips_dim, hidden_dim_features_gru=128, n_layers_gru=2):
        super(ThreeStreamFusionEncoder, self).__init__()
        # Define three independent Bi-GRUs
        self.hand_shape_gru = nn.GRU(hand_shape_dim, hidden_dim_features_gru, n_layers_gru, bidirectional=True, batch_first=True)
        self.hand_pos_gru   = nn.GRU(hand_pos_dim, hidden_dim_features_gru, n_layers_gru, bidirectional=True, batch_first=True)
        self.lips_gru       = nn.GRU(lips_dim, hidden_dim_features_gru, n_layers_gru, bidirectional=True, batch_first=True)
        
        # Fusion GRU: note the input size is 3 streams * 2 (bidirectional)
        self.fusion_gru = nn.GRU(hidden_dim_features_gru * 6, hidden_dim_features_gru * 2, n_layers_gru, bidirectional=True, batch_first=True)
        
    def forward(self, hand_shape, hand_pos, lips):
        hand_shape_out, _ = self.hand_shape_gru(hand_shape)  # (batch, seq, hidden_dim*2)
        hand_pos_out, _   = self.hand_pos_gru(hand_pos)
        lips_out, _       = self.lips_gru(lips)
        
        combined_features = torch.cat([hand_shape_out, hand_pos_out, lips_out], dim=-1)
        fusion_out, _ = self.fusion_gru(combined_features)
        return fusion_out  # Encoder output

class AttentionDecoder(nn.Module):
    def __init__(self, encoder_dim, output_dim, hidden_dim_decoder=512, n_layers=1):
        super(AttentionDecoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, hidden_dim_decoder)
        # GRU that takes concatenated [embedded, context] vectors.
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
    def __init__(self, hand_shape_dim, hand_pos_dim, lips_dim, output_dim, encoder_hidden_dim=128):
        super(JointCTCAttentionModel, self).__init__()
        self.encoder = ThreeStreamFusionEncoder(hand_shape_dim, hand_pos_dim, lips_dim, hidden_dim_features_gru=encoder_hidden_dim)
        self.attention_decoder = AttentionDecoder(encoder_dim=encoder_hidden_dim*4, output_dim=output_dim)
        # A fully connected layer for the CTC branch
        self.ctc_fc = nn.Linear(encoder_hidden_dim*4, output_dim)
    
    def forward(self, hand_shape, hand_pos, lips, target_seq=None):
        # Get encoder outputs
        encoder_out = self.encoder(hand_shape, hand_pos, lips)
        
        # CTC branch: predict directly from encoder outputs
        ctc_logits = self.ctc_fc(encoder_out)
        
        # Attention branch: only used during training (with teacher forcing)
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
            batch_y = batch_y.long().to(device)

            # Forward pass with teacher forcing using batch_y as target_seq
            ctc_logits, attn_logits = model(batch_X_hand_shape, batch_X_hand_pos, batch_X_lips, target_seq=batch_y)

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
        if epoch > 5000 and epoch <= 7000:
            alpha = 0.5
        elif epoch > 7000 and epoch <= 9000:
            alpha = 0.3
        else:
            alpha = 0.2
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
            batch_y = batch_y.long().to(device)
            
            # Forward pass with teacher forcing (pass batch_y as target_seq)
            ctc_logits, attn_logits = model(batch_X_hand_shape, batch_X_hand_pos, batch_X_lips, target_seq=batch_y)
            
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
        
        # Optionally, decode and compute PER or WER on the validation set.

        blank_token = phoneme_to_index["<UNK>"]
        decoded_val_sequences, true_val_sequences = decode_loader(model, val_loader, blank_token, index_to_phoneme, device)
        val_per = calculate_per_with_jiwer(decoded_val_sequences, true_val_sequences)
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {round(train_loss_avg, 3)}, " +
              f"Val Loss: {round(val_loss, 3)}, Accuracy (1-PER): {round(1 - val_per, 3)}, " +
              f"Time: {round(time.time() - epoch_start_time, 2)} sec")
        sys.stdout.flush()
        wandb.log({"val_per": 1 - val_per, "epoch": epoch + 1})
        wandb.log({"val_wer": val_per, "epoch": epoch + 1})
        if epoch % 10 == 0 and epoch > 2600:
            torch.save(model.state_dict(), f"/scratch2/bsow/Documents/ACSR/src/acsr/model2_epoch_{alpha}.pt")
    
    print("Training complete.")


def greedy_decoder(output, blank):
    arg_maxes = torch.argmax(output, dim=2)  # Get the most likely class for each time step
    decodes = []
    for args in arg_maxes:
        args = torch.unique_consecutive(args)  # Remove consecutive repeated indices
        decode = []
        for index in args:
            if index != blank:
                decode.append(index.item())  # Append non-blank and non-repeated tokens
        decodes.append(decode)
    return decodes

def viterbi_decoder(output, blank=0):
    """
    CTC Viterbi decoder for batched inputs.
    
    Args:
        output (np.ndarray or torch.Tensor): BxTxC matrix of log probabilities.
        blank (int): Index of the blank token.
    
    Returns:
        list: List of decoded sequences (collapsed) for each batch element.
    """
    # Convert PyTorch tensor to numpy if necessary
    if hasattr(output, "numpy"):
        output = output.numpy()
    
    # Handle batch dimension
    if output.ndim == 3:
        batch_size, T, C = output.shape
        decoded_sequences = []
        for b in range(batch_size):
            sequence = viterbi_decoder_single(output[b], blank)
            decoded_sequences.append(sequence)
        return decoded_sequences
    else:
        return viterbi_decoder_single(output, blank)

def viterbi_decoder_single(output_single, blank=0):
    """Viterbi decoding for a single sequence (TxC)."""
    T, C = output_single.shape
    labels = [c for c in range(C) if c != blank]
    
    delta_b = np.full(T, -np.inf)
    delta_c = {l: np.full(T, -np.inf) for l in labels}
    
    # Initialize first timestep
    delta_b[0] = output_single[0, blank]
    for l in labels:
        delta_c[l][0] = output_single[0, l]
    
    # Recurrence
    for t in range(1, T):
        # Update blank state
        prev_max_blank = delta_b[t-1] + output_single[t, blank]
        prev_max_label = max(delta_c[l][t-1] for l in labels) + output_single[t, blank]
        delta_b[t] = max(prev_max_blank, prev_max_label)
        
        # Update non-blank states
        for l in labels:
            option1 = delta_b[t-1] + output_single[t, l]
            option2 = delta_c[l][t-1] + output_single[t, l]
            delta_c[l][t] = max(option1, option2)
    
    # Backtracking
    best_path = []
    current_label = None
    if delta_b[-1] > max(delta_c[l][-1] for l in labels):
        current_state = 'blank'
    else:
        current_state = 'label'
        current_label = max(labels, key=lambda l: delta_c[l][-1])
    
    for t in reversed(range(T)):
        if current_state == 'blank':
            best_path.append(blank)
            if t > 0:
                if delta_b[t-1] > max(delta_c[l][t-1] for l in labels):
                    current_state = 'blank'
                else:
                    current_state = 'label'
                    current_label = max(labels, key=lambda l: delta_c[l][t-1])
        else:
            best_path.append(current_label)
            if delta_c[current_label][t] == delta_b[t-1] + output_single[t, current_label]:
                current_state = 'blank'
            else:
                current_state = 'label'
    
    best_path = best_path[::-1]  # Reverse to original order
    
    # Collapse the path
    collapsed = []
    prev_char = None
    for char in best_path:
        if char != blank:
            if char != prev_char:
                collapsed.append(char)
            prev_char = char
        else:
            prev_char = None
    
    return collapsed


def decode_loader(model, loader, blank, index_to_phoneme, device='cuda'):
    """
    Decode sequences from a DataLoader using the CTC branch of the joint model.
    
    Args:
        model: The JointCTCAttentionModel.
        loader: DataLoader that yields (hand_shape, hand_pos, lips, batch_y).
        blank: The blank token index used by CTC.
        index_to_phoneme: Dictionary mapping indices to phoneme strings.
        device: The torch device (e.g., 'cuda' or 'cpu').
    
    Returns:
        A tuple (all_decoded_sequences, all_true_sequences).
    """
    model.eval()  # Set the model to evaluation mode
    all_decoded_sequences = []
    all_true_sequences = []

    with torch.no_grad():  # Disable gradient computation
        for batch_X_hand_shape, batch_X_hand_pos, batch_X_lips, batch_y in loader:
            # Move data to device
            batch_X_hand_shape = batch_X_hand_shape.to(device)
            batch_X_hand_pos = batch_X_hand_pos.to(device)
            batch_X_lips = batch_X_lips.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass through the model (without teacher forcing)
            # This returns (ctc_logits, attn_logits); we use the CTC branch for decoding.
            ctc_logits, _ = model(batch_X_hand_shape, batch_X_hand_pos, batch_X_lips)
            
            # Use your greedy_decoder on the CTC logits.
            # Ensure that greedy_decoder expects logits of shape (batch, time, num_classes)
            decoded_phoneme_sequences = greedy_decoder(ctc_logits, blank=blank)
            decoded_phonemes = [
                [index_to_phoneme[idx] for idx in sequence] 
                for sequence in decoded_phoneme_sequences
            ]
            all_decoded_sequences.extend(decoded_phonemes)

            # Convert true labels (batch_y) to phoneme sequences.
            true_phoneme_sequences = []
            for sequence in batch_y:
                seq_phonemes = [
                    index_to_phoneme[idx.item()]
                    for idx in sequence 
                    if idx != blank and index_to_phoneme[idx.item()] != " "
                ]
                true_phoneme_sequences.append(seq_phonemes)
            all_true_sequences.extend(true_phoneme_sequences)

    return all_decoded_sequences, all_true_sequences

import jiwer


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

def beam_search_decode(
    cuedspeech_model,
    nextsyllable_model,
    inputs_am,
    blank_idx,
    index_to_syllable,
    beam_width=5,
    alpha=0.7,
    device="cuda",
    max_seq_len=15
):
    cuedspeech_model.eval()
    nextsyllable_model.eval()
    
    with torch.no_grad():
        inputs_am = [inp.to(device) for inp in inputs_am]
        logits_am = cuedspeech_model(*inputs_am)
        log_probs_am = torch.log_softmax(logits_am, dim=-1)
        batch_size, seq_len, vocab_size = log_probs_am.shape

        am_probs = log_probs_am[0, 0, :].cpu().numpy()
        predicted_token = np.argmax(am_probs)
        # Initialize beams with top tokens
        beams = []
        for token in top_tokens:
            beams.append({
                "sequence": [token],
                "scores": [am_probs[token]],
                "total_score": am_probs[token]
            })
        
        for t in range(1, seq_len):
            am_probs = log_probs_am[0, t, :].cpu().numpy()
            top_tokens = np.argsort(am_probs)[::-1][:beam_width]

            new_beams = []
            for beam in beams:
                for token in top_tokens:
                    am_prob = am_probs[token]
                    seq = beam["sequence"] 
                    # left pad sequence
                    padded_seq = ["<PAD>"]*(max_seq_len-len(seq)) + seq
                    lm_input = torch.tensor(
                        [s for s in padded_seq[-max_seq_len:]],
                        device=device
                    ).unsqueeze(0)
                    logits_lm = nextsyllable_model(lm_input)
                    predicted_token = torch.argmax(logits_lm, dim=-1).item()
            



        for t in range(seq_len):
            new_beams = defaultdict(lambda: {
                "score_blank": -np.inf,
                "score_non_blank": -np.inf
            })

            for beam in beams:
                # Current AM probabilities
                am_probs = log_probs_am[0, t, :].cpu().numpy()

                # Handle blank transition
                blank_score = np.logaddexp(
                    beam["score_blank"] + am_probs[blank_idx],
                    beam["score_non_blank"] + am_probs[blank_idx]
                )
                key = tuple(beam["sequence"])
                new_beams[key]["score_blank"] = np.logaddexp(
                    new_beams[key]["score_blank"],
                    blank_score
                )

                # Handle non-blank transitions
                for token in range(vocab_size):
                    if token == blank_idx:
                        continue

                    syllable = index_to_syllable[token]
                    new_seq = beam["sequence"].copy()
                    
                    # Apply CTC merge rule
                    if not new_seq or syllable != new_seq[-1]:
                        new_seq = new_seq + [syllable]
                    
                    # Get LM score
                    padded_seq = ["<PAD>"]*(max_seq_len-len(new_seq)) + new_seq
                    lm_input = torch.tensor(
                        [phoneme_to_index[s] for s in padded_seq[-max_seq_len:]],
                        device=device
                    ).unsqueeze(0)
                    
                    logits_lm = nextsyllable_model(lm_input)
                    lm_score = torch.log_softmax(logits_lm, dim=-1)[0, token].item()
                    
                    # Calculate scores
                    non_blank_score = am_probs[token] + alpha * lm_score
                    total_score = np.logaddexp(
                        beam["score_blank"] + non_blank_score,
                        beam["score_non_blank"] + non_blank_score
                        if new_seq == beam["sequence"] else -np.inf
                    )
                    
                    new_beams[tuple(new_seq)]["score_non_blank"] = np.logaddexp(
                        new_beams[tuple(new_seq)]["score_non_blank"],
                        total_score
                    )

            # Prune beams
            pruned_beams = []
            for key in new_beams:
                total_score = np.logaddexp(
                    new_beams[key]["score_blank"],
                    new_beams[key]["score_non_blank"]
                )
                pruned_beams.append({
                    "sequence": list(key),
                    "score_blank": new_beams[key]["score_blank"],
                    "score_non_blank": new_beams[key]["score_non_blank"],
                    "total_score": total_score
                })
            
            beams = sorted(pruned_beams, key=lambda x: x["total_score"], reverse=True)[:beam_width]

        # Return best sequence
        best_beam = max(beams, key=lambda x: x["total_score"])
        return [s for s in best_beam["sequence"] if s != "<PAD>"]
    

def decode_loader_combined(cuedspeech_model, nextsyllable_model, loader, 
                           blank, index_to_syllable, beam_width=5, alpha=0.7, device='cuda'):
    cuedspeech_model.eval()
    nextsyllable_model.eval()
    all_decoded_sequences = []
    all_true_sequences = []
    
    with torch.no_grad():
        for batch_X_hand_shape, batch_X_hand_pos, batch_X_lips, batch_y in loader:
            batch_size = batch_X_hand_shape.size(0)
            # Move data to device
            batch_X_hand_shape = batch_X_hand_shape.to(device)
            batch_X_hand_pos = batch_X_hand_pos.to(device)
            batch_X_lips = batch_X_lips.to(device)
            batch_y = batch_y.to(device)
            
            for i in range(batch_size):
                inputs_am = (batch_X_hand_shape[i].unsqueeze(0),
                             batch_X_hand_pos[i].unsqueeze(0),
                             batch_X_lips[i].unsqueeze(0))
                decoded_seq = beam_search_decode(
                    cuedspeech_model, nextsyllable_model, inputs_am, 
                    blank, index_to_syllable, beam_width, alpha, device
                )
                all_decoded_sequences.append(decoded_seq)
                
                # Process ground truth sequence
                true_seq_indices = batch_y[i]
                true_seq = [index_to_syllable[idx.item()] for idx in true_seq_indices if idx != blank]
                all_true_sequences.append(true_seq)
    
    return all_decoded_sequences, all_true_sequences



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
        #print("Batch X_teacher shape:", batch_X_teacher.shape)
        print("Batch y shape:", batch_y.shape)
        print("Output dim of the model: ", len(phoneme_to_index))
        break

    learning_rate = 1e-3
    batch_size = 16
    hidden_dim_fusion = 256
    epochs = 12000
    encoder_hidden_dim = 128
    output_dim = len(phoneme_to_index)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    level = "syllables"
    n_layers_gru = 2
    alpha = 0.4
    wandb.login(key="580ab03d7111ed25410f9831b06b544b5f5178a2")
    # Initialize W&B
    wandb.init(project="acsr", config={
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "encoder_hidden_dim": encoder_hidden_dim,
        "output_dim": output_dim,
        "n_layers_gru": n_layers_gru,
        "alpha": alpha,
        "device": device,
        "level": level,
    })

    # Define the model
    acoustic_model = JointCTCAttentionModel(
        hand_shape_dim=19,  # 21 keypoints × 3 coordinates
        hand_pos_dim=30,    # 3 coordinates (x, y, z)
        lips_dim=10,       # 40 keypoints × 3 coordinates
        output_dim=output_dim,  # Number of phonemes
        encoder_hidden_dim=encoder_hidden_dim,  # Hidden dimension for GRUs
    )

    # Optimizer
    optimizer = Adam(acoustic_model.parameters(), lr=1e-3, weight_decay=1e-5)
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    print("Training on device:", device)
    acoustic_model.to(device)
    # load the trained acoustic model
    #acoustic_model.load_state_dict(torch.load("/scratch2/bsow/Documents/ACSR/output/saved_models/acoustic_model2.pth", map_location=torch.device('cpu')))
    acoustic_model.to(device)
    
    # Start training
    train_model(acoustic_model, train_loader, val_loader, num_epochs=epochs, alpha=alpha, device=device, optimizer=optimizer)
    torch.save(acoustic_model.state_dict(), "/scratch2/bsow/Documents/ACSR/output/saved_models/acoustic_model4.pth")


    blank_token =  phoneme_to_index["<UNK>"]
    decoded_train_sequences, true_train_sequences = decode_loader(acoustic_model, train_loader, blank_token, index_to_phoneme, device)
    decoded_val_sequences, true_val_sequences = decode_loader(acoustic_model, val_loader, blank_token, index_to_phoneme, device)

    # Print results
    #print("Decoded training phoneme sequences:", decoded_train_sequences[:5])
    #print("True training phoneme sequences:", true_train_sequences[:5])
    print("Decoded validation phoneme sequences:", decoded_val_sequences)
    print("True validation phoneme sequences:", true_val_sequences)
    sys.stdout.flush()
    train_per = calculate_per_with_jiwer(decoded_train_sequences, true_train_sequences)
    val_per = calculate_per_with_jiwer(decoded_val_sequences, true_val_sequences)
    print("Training PER (jiwer):", train_per, "1 - PER: ", 1 - train_per)
    print("Validation PER (jiwer):", val_per, "1 - PER: ", 1 - val_per)
    sys.stdout.flush()
    
    # Save the trained acoustic model
    print("Acoustic model saved.")
    print("="*100)
    # Log the model as a W&B artifact
    #model_artifact = wandb.Artifact("acoustic_model", type="model")
    #model_artifact.add_file("/scratch2/bsow/Documents/ACSR/output/saved_models/acoustic_model.pth")
    #wandb.log_artifact(model_artifact)

    # Initialize model
    #nextsyllable_model = NextSyllableLSTM(
    #    vocab_size=len(phoneme_to_index),
    #    embedding_dim=200,
    #    hidden_dim=512,
    #    num_layers=4,
    #    dropout=0.2
    #).to(device)
###
    # Load model weights
    #nextsyllable_model.load_state_dict(torch.load("/scratch2/bsow/Documents/ACSR/src/acsr/wandb/run-20250131_113223-rge6w8nh/files/best_syllable_model_def2.pth", map_location=torch.device('cpu')))
    ## Ensure both models are on the same device
    #nextsyllable_model.to(device)
    #nextsyllable_model.eval()
    ##
    ## After training your models, perform decoding
    #blank_token = phoneme_to_index["<UNK>"]
    #beam_width = 6
    #alpha = 0.5  # Adjust alpha to balance between models
    #
    #decoded_val_sequences, true_val_sequences = decode_loader_combined(
    #    acoustic_model, nextsyllable_model, val_loader,
    #    blank_token, index_to_phoneme, beam_width, alpha, device
    #)
    #print("Decoded validation syllable sequences:", decoded_val_sequences)
    #print("True validation syllable sequences:", true_val_sequences)
    #sys.stdout.flush()
##
    ## Evaluate performance
    #train_per_beam = calculate_per_with_jiwer(decoded_train_sequences, true_train_sequences)
    #val_per_beam = calculate_per_with_jiwer(decoded_val_sequences, true_val_sequences)
    #print("Training PER (jiwer) after combining models:", train_per_beam, "1 - PER: ", 1 - train_per_beam)
    #print("Validation PER (jiwer) after combining models:", val_per_beam, "1 - PER: ", 1 - val_per_beam)


    #Finish W&B run
    wandb.finish()

    