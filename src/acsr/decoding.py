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
import wandb  # Import W&B



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

# Load phoneme-to-index mapping
# with open(r"/scratch2/bsow/Documents/ACSR/data/training_videos/CSF22_train/phonelist.csv", "r") as file:
with open(r"/scratch2/bsow/Documents/ACSR/data/training_videos/syllables.txt", "r") as file:
    reader = csv.reader(file)
    vocabulary_list = [row[0] for row in reader]

phoneme_to_index = {phoneme: idx for idx, phoneme in enumerate(set(list(vocabulary_list)))}
phoneme_to_index[" "] = len(phoneme_to_index)
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


def prepare_data_for_videos_no_sliding_windows(base_names, phoneme_files, features_dir, labels_dir, phoneme_to_index):
    all_videos_data = {}
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
            X_student_hand_shape = features_df[hand_shape_columns].to_numpy()
            X_student_hand_pos = features_df[hand_pos_columns].to_numpy()
            X_student_lips = features_df[lip_columns].to_numpy()

            # Load phoneme labels from CSV file
            labels_path = os.path.join(labels_dir, f"{base_name}.csv")
            if not os.path.exists(labels_path):
                raise FileNotFoundError(f"Phoneme label file not found: {labels_path}")
            
            # Read the CSV file
            phoneme_labels = pd.read_csv(labels_path, header=None).squeeze().tolist()  # Convert to list of phonemes

            # Convert phoneme labels to syllables
            syllable_labels = syllabify_ipa(" ".join(phoneme_labels))

            # Convert syllable labels to indices
            syllable_indices = []
            for syllable in syllable_labels:
                if syllable not in phoneme_to_index:
                    raise ValueError(f"Syllable '{syllable}' not found in the vocabulary. File: {base_name}")
                syllable_indices.append(phoneme_to_index[syllable])

            # Combine features and syllable indices
            all_videos_data[base_name] = {
                "X_student_hand_shape": X_student_hand_shape,  # Hand shape features
                "X_student_hand_pos": X_student_hand_pos,      # Hand position features
                "X_student_lips": X_student_lips,              # Lip features
                "y": syllable_indices,                         # Syllable labels (sequence)
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
    # Get the number of samples
    num_samples = len(data['X_student_hand_shape'])
    split_idx = int(num_samples * train_ratio)

    # Randomize the data
    indices = np.random.permutation(num_samples)

    # Split hand shape features
    X_student_hand_shape = [data['X_student_hand_shape'][i] for i in indices]
    X_student_hand_shape_train = X_student_hand_shape[:split_idx]
    X_student_hand_shape_val = X_student_hand_shape[split_idx:]

    # Split hand position features
    X_student_hand_pos = [data['X_student_hand_pos'][i] for i in indices]
    X_student_hand_pos_train = X_student_hand_pos[:split_idx]
    X_student_hand_pos_val = X_student_hand_pos[split_idx:]

    # Split lip features
    X_student_lips = [data['X_student_lips'][i] for i in indices]
    X_student_lips_train = X_student_lips[:split_idx]
    X_student_lips_val = X_student_lips[split_idx:]

    # Split labels
    y = [data['y'][i] for i in indices]
    y_train = y[:split_idx]
    y_val = y[split_idx:]

    # Create train and validation data dictionaries
    train_data = {
        'X_student_hand_shape': X_student_hand_shape_train,
        'X_student_hand_pos': X_student_hand_pos_train,
        'X_student_lips': X_student_lips_train,
        'y': y_train
    }
    val_data = {
        'X_student_hand_shape': X_student_hand_shape_val,
        'X_student_hand_pos': X_student_hand_pos_val,
        'X_student_lips': X_student_lips_val,
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

    # Pad labels with the padding token (e.g., phoneme_to_index[" "])
    labels_padded = pad_sequence([torch.tensor(y, dtype=torch.float32) for y in labels], batch_first=True, padding_value=phoneme_to_index[" "])

    return hand_shape_padded, hand_pos_padded, lips_padded, labels_padded

def data_to_dataloader(data, batch_size=4, shuffle=True):
    # Create a dataset from the lists
    dataset = list(zip(data['X_student_hand_shape'], data['X_student_hand_pos'], data['X_student_lips'], data['y']))

    # Create a DataLoader with the custom collate function
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=custom_collate_fn  # Use the custom collate function
    )
    return dataloader


# Student model
class ThreeStreamFusionModel(nn.Module):
    def __init__(self, hand_shape_dim, hand_pos_dim, lips_dim, output_dim, hidden_dim_features_gru=128, hidden_dim_fusion_gru=256):
        super(ThreeStreamFusionModel, self).__init__()
        
        # Bi-GRU layers for each stream
        self.hand_shape_gru = nn.GRU(
            input_size=hand_shape_dim,  # Directly use hand_shape_dim
            hidden_size=hidden_dim_features_gru,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        
        self.hand_pos_gru = nn.GRU(
            input_size=hand_pos_dim,  # Directly use hand_pos_dim
            hidden_size=hidden_dim_features_gru,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        
        self.lips_gru = nn.GRU(
            input_size=lips_dim,  # Directly use lips_dim
            hidden_size=hidden_dim_features_gru,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        
        # Second Bi-GRU layer after concatenation
        self.fusion_gru = nn.GRU(
            input_size= hidden_dim_features_gru * 6,  # 3 streams * 2 (bidirectional)
            hidden_size=hidden_dim_fusion_gru,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        
        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim_fusion_gru*2, output_dim),
        )

    def forward(self, hand_shape, hand_pos, lips):
        # Get batch size and sequence length
        batch_size, seq_len, _ = hand_shape.shape
        
        # Pass through stream-wise Bi-GRUs
        hand_shape_out, _ = self.hand_shape_gru(hand_shape)  # (batch_size, seq_len, hidden_dim * 2)
        hand_pos_out, _ = self.hand_pos_gru(hand_pos)        # (batch_size, seq_len, hidden_dim * 2)
        lips_out, _ = self.lips_gru(lips)                    # (batch_size, seq_len, hidden_dim * 2)
        
        # Concatenate the outputs of the three streams
        combined_features = torch.cat([hand_shape_out, hand_pos_out, lips_out], dim=-1)  # (batch_size, seq_len, hidden_dim * 6)
        
        # Pass through the fusion Bi-GRU
        fusion_out, _ = self.fusion_gru(combined_features)  # (batch_size, seq_len, hidden_dim * 4)
        
        # Final predictions
        output = self.fc(fusion_out)  # (batch_size, seq_len, output_dim)
        return output


import sys 

def sequence_level_distillation_loss(student_logits, teacher_logits, batch_y, input_lengths, label_lengths, device):
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
            
            # Compute input_lengths
            input_lengths = torch.full(
                (batch_X_student_hand_shape.size(0),),  # Batch size
                student_logits.size(1),  # Sequence length (time steps) from student_logits
                dtype=torch.long,
                device=device
            )
            # Compute label_lengths (excluding padding)
            label_lengths = (batch_y != phoneme_to_index[" "]).sum(dim=1).to(device)
            
            # Compute loss
            loss = sequence_level_distillation_loss(student_logits, None, batch_y, input_lengths, label_lengths, device)
            
            # Accumulate loss
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss

import time

def train_student_model(student_model, teacher_model, train_loader, val_loader, num_epochs=50, device="cuda"):
    # Set teacher model to evaluation mode
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
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
            label_lengths = (batch_y != phoneme_to_index[" "]).sum(dim=1).to(device)

            # Compute loss
            loss = sequence_level_distillation_loss(student_logits, teacher_logits, batch_y, input_lengths, label_lengths, device)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Log training loss to W&B
        wandb.log({"train_loss": epoch_loss / len(train_loader), "epoch": epoch + 1})

        # Evaluate the model on the validation set
        val_loss = validate_student_model(student_model, val_loader, device)
        
        # Log validation loss to W&B
        wandb.log({"val_loss": val_loss, "epoch": epoch + 1})
        
        # Decode validation sequences and calculate PER
        blank_token = phoneme_to_index[" "]
        decoded_val_sequences, true_val_sequences = decode_loader(student_model, val_loader, blank_token, index_to_phoneme)
        val_per = calculate_per_with_jiwer(decoded_val_sequences, true_val_sequences)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {round(epoch_loss / len(train_loader), 3)}, Validation Loss: {round(val_loss, 3)}, Accuracy (1 - PER): {round(1 - val_per, 3)}, Time: {round(time.time() - epoch_start_time, 2)} seconds")
        sys.stdout.flush()
        
        # Log validation PER to W&B
        wandb.log({"val_per": 1-val_per, "epoch": epoch + 1})
        wandb.log({"val_wer": val_per, "epoch": epoch + 1})

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
    
    print("Training complete.")


def greedy_decoder(output, blank):
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

import torch
import torch.nn.functional as F
from collections import defaultdict
import heapq

def compute_log_probs(logits):
    """
    Converts logits to log probabilities.
    Args:
        logits (torch.Tensor): Output logits from the model.
    Returns:
        torch.Tensor: Log probabilities.
    """
    return F.log_softmax(logits, dim=-1)

def combine_log_probs(log_probs_am, log_probs_lm, alpha=0.7):
    """
    Combines acoustic and language model log probabilities using a weighted sum.
    Args:
        log_probs_am (torch.Tensor): Log probabilities from the acoustic model.
        log_probs_lm (torch.Tensor): Log probabilities from the language model.
        alpha (float): Weight for the acoustic model probabilities.
    Returns:
        torch.Tensor: Combined log probabilities.
    """
    combined_log_probs = alpha * log_probs_am + (1 - alpha) * log_probs_lm
    return combined_log_probs

def beam_search_decode(cuedspeech_model, nextsyllable_model, inputs_am, 
                       blank, index_to_syllable, beam_width=5, alpha=0.7, device='cuda'):
    """
    Performs beam search decoding using both the cuedspeech and next syllable models.
    Args:
        cuedspeech_model: The trained cuedspeech (acoustic) model.
        nextsyllable_model: The trained next syllable (language) model.
        inputs_am (tuple): Tuple of inputs for the cuedspeech model.
        blank (int): Index of the blank symbol.
        index_to_syllable (dict): Mapping from indices to syllables.
        beam_width (int): The beam width for beam search.
        alpha (float): Weight for the acoustic model log probabilities.
        device (str): Device to run the models on.
    Returns:
        list: Decoded syllable sequence.
    """
    cuedspeech_model.eval()
    nextsyllable_model.eval()
    with torch.no_grad():
        # Move inputs to device
        inputs_am = [inp.to(device) for inp in inputs_am]
        
        # Get acoustic model outputs
        logits_am = cuedspeech_model(*inputs_am)  # Shape: [batch_size, seq_len, vocab_size]
        log_probs_am = compute_log_probs(logits_am)  # Log probabilities from acoustic model
        
        batch_size, seq_len, vocab_size = log_probs_am.size()
        
        # Initialize beams
        beams = [([], 0.0)]  # Each beam is a tuple (sequence, score)
        
        for t in range(seq_len):
            new_beams = []
            log_probs_am_t = log_probs_am[:, t, :]  # Shape: [batch_size, vocab_size]
            
            for seq, score in beams:
                # Prepare input for language model
                if seq:
                    lm_input = torch.tensor([phoneme_to_index[seq[-1]]], device=device).unsqueeze(0)
                else:
                    lm_input = torch.tensor([[blank]], device=device)
                
                # Get language model log probabilities
                logits_lm = nextsyllable_model(lm_input)  # Shape: [batch_size, vocab_size]
                log_probs_lm = compute_log_probs(logits_lm)
                log_probs_lm = log_probs_lm.squeeze(0)  # Shape: [vocab_size]
                
                # Combine log probabilities
                combined_log_probs = combine_log_probs(log_probs_am_t.squeeze(0), log_probs_lm, alpha=alpha)
                
                # Get top K candidates
                topk_log_probs, topk_indices = torch.topk(combined_log_probs, beam_width)
                
                for i in range(beam_width):
                    idx = topk_indices[i].item()
                    log_prob = topk_log_probs[i].item()
                    new_seq = seq.copy()
                    if idx != blank:
                        new_seq.append(index_to_syllable[idx])
                    new_score = score + log_prob
                    new_beams.append((new_seq, new_score))
            
            # Keep top K beams
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
        
        # Return the best sequence
        best_seq, _ = beams[0]
        return best_seq

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
    all_videos_data = prepare_data_for_videos_no_sliding_windows(
        base_names, phoneme_files, features_dir, labels_dir, phoneme_to_index
    )

    # Final organized data
    data = {
        "X_student_hand_shape": [all_videos_data[video]["X_student_hand_shape"] for video in all_videos_data],  # Hand shape coordinates
        "X_student_hand_pos": [all_videos_data[video]["X_student_hand_pos"] for video in all_videos_data],      # Hand position coordinates
        "X_student_lips": [all_videos_data[video]["X_student_lips"] for video in all_videos_data],              # Lip coordinates
        "y": [all_videos_data[video]["y"] for video in all_videos_data],                                        # Phoneme labels
    }
    # Split data
    train_data, val_data = train_val_split(data)

    # Prepare DataLoaders
    train_loader = data_to_dataloader(train_data, batch_size=16, shuffle=True)
    val_loader = data_to_dataloader(val_data, batch_size=16, shuffle=False)

    print("Len of train dataset", len(train_data['X_student_hand_shape']))
    print("Len of val dataset", len(val_data['X_student_hand_shape']))

    # Check the DataLoader output
    for batch_X_student_hand_shape, batch_X_student_hand_pos, batch_X_student_lips, batch_y in train_loader:
        print("Batch X_student_hand_shape shape:", batch_X_student_hand_shape.shape)
        print("Batch X_student_hand_pos shape:", batch_X_student_hand_pos.shape)
        print("Batch X_student_lips shape:", batch_X_student_lips.shape)
        #print("Batch X_teacher shape:", batch_X_teacher.shape)
        print("Batch y shape:", batch_y.shape)
        print("Output dim of the model: ", len(phoneme_to_index))
        break
    wandb.login(key="580ab03d7111ed25410f9831b06b544b5f5178a2")
    # Initialize W&B
    wandb.init(project="acsr", config={
        "learning_rate": 1e-3,
        "batch_size": 16,
        "epochs": 500,
        "hidden_dim_fusion": 256,
        "hidden_dim_features": 128,
        "output_dim": len(phoneme_to_index),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "level": "syllables"
    })

    # Define the model
    student_model = ThreeStreamFusionModel(
        hand_shape_dim=19,  # 21 keypoints × 3 coordinates
        hand_pos_dim=30,    # 3 coordinates (x, y, z)
        lips_dim=10,       # 40 keypoints × 3 coordinates
        output_dim=len(phoneme_to_index),  # Number of phonemes
        hidden_dim_features_gru=128,  # Hidden dimension for GRUs
        hidden_dim_fusion_gru=256,     # Hidden dimension for GRUs
    )

    # Optimizer
    optimizer = Adam(student_model.parameters(), lr=1e-3, weight_decay=1e-5)
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    print("Training on device:", device)
    student_model.to(device)
    
    # Start training
    train_student_model(student_model, None, train_loader, val_loader, num_epochs=500, device=device)

    blank_token =  phoneme_to_index[" "]
    decoded_train_sequences, true_train_sequences = decode_loader(student_model, train_loader, blank_token, index_to_phoneme)
    decoded_val_sequences, true_val_sequences = decode_loader(student_model, val_loader, blank_token, index_to_phoneme)

    # Print results
    print("Decoded training phoneme sequences:", decoded_train_sequences[:5])
    print("True training phoneme sequences:", true_train_sequences[:5])
    print("Decoded validation phoneme sequences:", decoded_val_sequences[:5])
    print("True validation phoneme sequences:", true_val_sequences[:5])
    sys.stdout.flush()
    train_per = calculate_per_with_jiwer(decoded_train_sequences, true_train_sequences)
    val_per = calculate_per_with_jiwer(decoded_val_sequences, true_val_sequences)
    print("Training PER (jiwer):", train_per, "1 - PER: ", 1 - train_per)
    print("Validation PER (jiwer):", val_per, "1 - PER: ", 1 - val_per)
    sys.stdout.flush()
    
    # Save the trained student model
    torch.save(student_model.state_dict(), "/scratch2/bsow/Documents/ACSR/output/saved_models/student_model.pth")
    print("Student model saved.")
    
    # Log the model as a W&B artifact
    model_artifact = wandb.Artifact("student_model", type="model")
    model_artifact.add_file("/scratch2/bsow/Documents/ACSR/output/saved_models/student_model.pth")
    wandb.log_artifact(model_artifact)
    
    ## load the next syllable model
    #from syllabification.py import *
#
    ## Initialize model
    #nextsyllable_model = NextSyllableLSTM(
    #    vocab_size=len(phoneme_to_index),
    #    embedding_dim=256,
    #    hidden_dim=512,
    #    num_layers=2
    #)
#
    ## Load model weights
    #nextsyllable_model.load_state_dict(torch.load("/scratch2/bsow/Documents/ACSR/output/saved_models/next_syllable_lstm.pth"))
    ## Ensure both models are on the same device
    #nextsyllable_model.to(device)
    #
    ## After training your models, perform decoding
    #blank_token = phoneme_to_index[" "]
    #beam_width = 5
    #alpha = 0.7  # Adjust alpha to balance between models
    #
    #decoded_train_sequences, true_train_sequences = decode_loader_combined(
    #    student_model, nextsyllable_model, train_loader,
    #    blank_token, index_to_phoneme, dataset, beam_width, alpha, device
    #)
    #decoded_val_sequences, true_val_sequences = decode_loader_combined(
    #    student_model, nextsyllable_model, val_loader,
    #    blank_token, index_to_phoneme, dataset, beam_width, alpha, device
    #)
    #
    ## Evaluate performance
    #train_per_beam = calculate_per_with_jiwer(decoded_train_sequences, true_train_sequences)
    #val_per_beam = calculate_per_with_jiwer(decoded_val_sequences, true_val_sequences)
    #print("Training PER (jiwer) after combining models:", train_per_beam)
    #print("Validation PER (jiwer) after combining models:", val_per_beam)


    # Finish W&B run
    wandb.finish()