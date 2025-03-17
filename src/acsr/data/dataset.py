"""
Dataset module for the ACSR system.
This module contains dataset-related classes and functions.
"""

import os
import csv
import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import Counter

from ..utils.text_processing import syllabify_ipa


class SyllableDataset(Dataset):
    """
    Dataset class for syllable sequences.
    
    Args:
        sequences (list): List of (input_seq, target) tuples.
        syllable_to_idx (dict): Mapping from syllables to indices.
    """
    
    def __init__(self, sequences, syllable_to_idx):
        self.sequences = sequences
        self.syllable_to_idx = syllable_to_idx
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        input_seq, target = self.sequences[idx]
        input_indices = [self.syllable_to_idx.get(syl, self.syllable_to_idx["<UNK>"]) 
                        for syl in input_seq]
        target_idx = self.syllable_to_idx.get(target, self.syllable_to_idx["<UNK>"])
        return torch.tensor(input_indices), torch.tensor(target_idx)


def load_csv_files(directory, filename_pattern):
    """
    Load CSV files from a directory based on a filename pattern.
    
    Args:
        directory (str): Directory containing CSV files.
        filename_pattern (str): Pattern to match in filenames.
        
    Returns:
        dict: Dictionary mapping base names to DataFrames.
    """
    files_data = {}
    for filename in os.listdir(directory):
        if filename_pattern in filename:
            df = pd.read_csv(os.path.join(directory, filename))
            df.dropna(inplace=True)
            base_name = filename.split(filename_pattern)[0]
            if "sent_" not in base_name:
                continue
            files_data[base_name] = df
    return files_data


def find_phoneme_files(directory, base_names):
    """
    Find corresponding phoneme files based on the base names of position filenames.
    
    Args:
        directory (str): Directory containing phoneme files.
        base_names (list): List of base names to search for.
        
    Returns:
        dict: Dictionary mapping base names to phoneme file paths.
    """
    phoneme_files = {}
    for base_name in base_names:
        phoneme_file = os.path.join(directory, f'{base_name}.csv')
        if os.path.exists(phoneme_file):
            phoneme_files[base_name] = phoneme_file
    return phoneme_files


def unique(sequence):
    """
    Return a list of unique elements while preserving order.
    
    Args:
        sequence (list): Input sequence.
        
    Returns:
        list: List of unique elements in the order they first appear.
    """
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


def load_features(directory, base_name):
    """
    Load features from a CSV file.
    
    Args:
        directory (str): Directory containing feature files.
        base_name (str): Base name of the feature file.
        
    Returns:
        pd.DataFrame: DataFrame containing features.
    """
    file_path = os.path.join(directory, f"{base_name}_features.csv")
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)  # Drop rows with NaN values
    return df


def prepare_data_for_videos_no_sliding_windows(base_names, phoneme_files, features_dir, labels_dir, 
                                              phoneme_to_index, img_size=(28, 28),
                                              lips_dir=None):
    """
    Prepare data for videos without using sliding windows.
    
    Args:
        base_names (list): List of base names.
        phoneme_files (dict): Dictionary mapping base names to phoneme file paths.
        features_dir (str): Directory containing feature files.
        labels_dir (str): Directory containing label files.
        phoneme_to_index (dict): Mapping from phonemes to indices.
        img_size (tuple, optional): Size of images. Defaults to (28, 28).
        lips_dir (str, optional): Directory containing lip ROI videos. Defaults to None.
        
    Returns:
        tuple: (train_videos_data, syllable_counter)
    """
    train_videos_data = {}
    syllable_counter = Counter()
    
    for base_name in sorted(base_names):
        if base_name in phoneme_files:
            # Load pre-extracted features
            features_df = load_features(features_dir, base_name)
            if 'frame_number' not in features_df.columns:
                raise ValueError(f"Feature file for {base_name} does not contain 'frame_number' column.")

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
            train_videos_data[base_name] = {
                "X_acoustic_hand_shape": X_acoustic_hand_shape,
                "X_acoustic_hand_pos": X_acoustic_hand_pos,
                "X_acoustic_lips": X_acoustic_lips,
                "X_visual_lips": None,  # Visual modality (if needed)
                "y": syllable_indices,
            }
    
    return train_videos_data, syllable_counter 