"""
Main entry point for the ACSR system.
This module contains the main function to run the system.
"""

import os
import csv
import sys
import time
import pandas as pd
import torch
from torch.optim import Adam
import wandb

from .config import (
    ACOUSTIC_MODEL_PARAMS, DEVICE, FEATURES_DIR_TRAIN, FEATURES_DIR_TEST,
    TRAIN_LABELS_DIR, TEST_LABELS_DIR, VOCAB_FILE, MODELS_DIR
)
from .models import JointCTCAttentionModel, NextSyllableLSTM
from .data import (
    load_csv_files, find_phoneme_files, prepare_data_for_videos_no_sliding_windows,
    data_to_dataloader
)
from .train import train_model
from .decode import decode_loader, decode_loader_with_rescoring
from .utils.metrics import calculate_per_with_jiwer, compute_and_print_wer_complements
from .utils.text_processing import syllables_to_phonemes


def load_vocabulary(vocab_file):
    """
    Load vocabulary from a file.
    
    Args:
        vocab_file (str): Path to the vocabulary file.
        
    Returns:
        tuple: (phoneme_to_index, index_to_phoneme)
    """
    with open(vocab_file, "r") as file:
        reader = csv.reader(file)
        vocabulary_list = [row[0] for row in reader]
    
    # Create unique vocabulary
    seen = set()
    unique_vocab = [x for x in vocabulary_list if not (x in seen or seen.add(x))]
    
    # Create mappings
    phoneme_to_index = {phoneme: idx for idx, phoneme in enumerate(unique_vocab)}
    index_to_phoneme = {idx: phoneme for phoneme, idx in phoneme_to_index.items()}
    
    return phoneme_to_index, index_to_phoneme


def prepare_data():
    """
    Prepare data for training and evaluation.
    
    Returns:
        tuple: (train_loader, val_loader, phoneme_to_index, index_to_phoneme)
    """
    # Load vocabulary
    phoneme_to_index, index_to_phoneme = load_vocabulary(VOCAB_FILE)
    
    # Load features
    features_data_train = load_csv_files(FEATURES_DIR_TRAIN, "_features")
    features_data_test = load_csv_files(FEATURES_DIR_TEST, "_features")
    
    # Find phoneme files
    base_names_train = features_data_train.keys()
    base_names_test = features_data_test.keys()
    phoneme_files_train = find_phoneme_files(TRAIN_LABELS_DIR, base_names_train)
    phoneme_files_test = find_phoneme_files(TEST_LABELS_DIR, base_names_test)
    
    print("Number of phoneme files found in the train set:", len(phoneme_files_train))
    print("Number of phoneme files found in the test set:", len(phoneme_files_test))
    
    # Prepare data
    train_videos_data, syllable_counter = prepare_data_for_videos_no_sliding_windows(
        base_names_train, phoneme_files_train, FEATURES_DIR_TRAIN, TRAIN_LABELS_DIR, phoneme_to_index
    )
    
    test_videos_data, _ = prepare_data_for_videos_no_sliding_windows(
        base_names_test, phoneme_files_test, FEATURES_DIR_TEST, TEST_LABELS_DIR, phoneme_to_index
    )
    
    # Save syllable distribution
    syllable_df = pd.DataFrame.from_dict(syllable_counter, orient='index', columns=['frequency'])
    syllable_df.index.name = 'syllable'
    syllable_df.reset_index(inplace=True)
    syllable_df = syllable_df.sort_values(by='frequency', ascending=False)
    output_csv_path = os.path.join(os.path.dirname(__file__), 'syllable_distribution.csv')
    syllable_df.to_csv(output_csv_path, index=False)
    print(f"Syllable distribution saved to {output_csv_path}")
    
    # Organize data
    train_data = {
        "X_acoustic_hand_shape": [train_videos_data[video]["X_acoustic_hand_shape"] for video in train_videos_data],
        "X_acoustic_hand_pos": [train_videos_data[video]["X_acoustic_hand_pos"] for video in train_videos_data],
        "X_acoustic_lips": [train_videos_data[video]["X_acoustic_lips"] for video in train_videos_data],
        "X_visual_lips": [],
        "y": [train_videos_data[video]["y"] for video in train_videos_data],
    }
    
    test_data = {
        "X_acoustic_hand_shape": [test_videos_data[video]["X_acoustic_hand_shape"] for video in test_videos_data],
        "X_acoustic_hand_pos": [test_videos_data[video]["X_acoustic_hand_pos"] for video in test_videos_data],
        "X_acoustic_lips": [test_videos_data[video]["X_acoustic_lips"] for video in test_videos_data],
        "X_visual_lips": [],
        "y": [test_videos_data[video]["y"] for video in test_videos_data],
    }
    
    # Create DataLoaders
    train_loader = data_to_dataloader(train_data, batch_size=ACOUSTIC_MODEL_PARAMS["batch_size"], shuffle=True)
    val_loader = data_to_dataloader(test_data, batch_size=ACOUSTIC_MODEL_PARAMS["batch_size"], shuffle=False)
    
    print("Len of train dataset:", len(train_data['X_acoustic_hand_shape']))
    print("Len of val dataset:", len(test_data['X_acoustic_hand_shape']))
    
    return train_loader, val_loader, phoneme_to_index, index_to_phoneme


def train_acoustic_model(train_loader, val_loader, phoneme_to_index, index_to_phoneme):
    """
    Train the acoustic model.
    
    Args:
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        phoneme_to_index (dict): Mapping from phonemes to indices.
        index_to_phoneme (dict): Mapping from indices to phonemes.
        
    Returns:
        JointCTCAttentionModel: Trained model.
    """
    # Get sample batch to determine dimensions
    for batch_X_acoustic_hand_shape, batch_X_acoustic_hand_pos, batch_X_acoustic_lips, batch_y in train_loader:
        print("Batch X_acoustic_hand_shape shape:", batch_X_acoustic_hand_shape.shape)
        print("Batch X_acoustic_hand_pos shape:", batch_X_acoustic_hand_pos.shape)
        print("Batch X_acoustic_lips shape:", batch_X_acoustic_lips.shape)
        print("Batch y shape:", batch_y.shape)
        print("Output dim of the model:", len(phoneme_to_index))
        break
    
    # Initialize model
    model = JointCTCAttentionModel(
        hand_shape_dim=batch_X_acoustic_hand_shape.shape[-1],
        hand_pos_dim=batch_X_acoustic_hand_pos.shape[-1],
        lips_dim=batch_X_acoustic_lips.shape[-1],
        visual_lips_dim=None,
        output_dim=len(phoneme_to_index),
        hidden_dim=ACOUSTIC_MODEL_PARAMS["encoder_hidden_dim"],
    )
    
    # Optimizer
    optimizer = Adam(model.parameters(), lr=ACOUSTIC_MODEL_PARAMS["learning_rate"], weight_decay=1e-5)
    
    # Move model to device
    model.to(DEVICE)
    
    # Initialize W&B
    wandb.init(project="acsr", config=ACOUSTIC_MODEL_PARAMS)
    
    # Train model
    model_save_path = os.path.join(MODELS_DIR, "model_decoding_last_new_features.pt")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=ACOUSTIC_MODEL_PARAMS["epochs"],
        alpha=ACOUSTIC_MODEL_PARAMS["alpha"],
        device=DEVICE,
        blank_idx=phoneme_to_index["<UNK>"],
        index_to_phoneme=index_to_phoneme,
        model_save_path=model_save_path
    )
    
    return model


def evaluate_model(model, val_loader, phoneme_to_index, index_to_phoneme):
    """
    Evaluate the model on the validation set.
    
    Args:
        model (nn.Module): Model to evaluate.
        val_loader (DataLoader): Validation data loader.
        phoneme_to_index (dict): Mapping from phonemes to indices.
        index_to_phoneme (dict): Mapping from indices to phonemes.
    """
    blank_token = phoneme_to_index["<UNK>"]
    decoded_val_sequences, true_val_sequences, decoded_val_gestures, true_val_gestures = decode_loader(
        model, val_loader, blank_token, index_to_phoneme, DEVICE, training=False
    )
    
    # Print results
    print("Decoded validation phoneme sequences:", decoded_val_sequences[-5:])
    print("True validation phoneme sequences:", true_val_sequences[-5:])
    
    # Calculate PER for syllables
    val_per = calculate_per_with_jiwer(decoded_val_sequences, true_val_sequences)
    print("Validation PER (jiwer):", val_per, "1 - PER:", 1 - val_per)
    compute_and_print_wer_complements(decoded_val_sequences, true_val_sequences)
    
    # Calculate PER for gestures
    print("Decoded validation gesture sequences:", decoded_val_gestures[-5:])
    print("True validation gesture sequences:", true_val_gestures[-5:])
    val_per_gestures = calculate_per_with_jiwer(decoded_val_gestures, true_val_gestures)
    print("Validation PER (jiwer) for gestures:", val_per_gestures, "1 - PER:", 1 - val_per_gestures)
    compute_and_print_wer_complements(decoded_val_gestures, true_val_gestures)
    
    # Calculate PER for phonemes
    decoded_val_characters = [syllables_to_phonemes(seq) for seq in decoded_val_sequences]
    true_val_characters = [syllables_to_phonemes(seq) for seq in true_val_sequences]
    val_per_characters = calculate_per_with_jiwer(decoded_val_characters, true_val_characters)
    print("Validation PER (jiwer) for characters:", val_per_characters, "1 - PER:", 1 - val_per_characters)
    compute_and_print_wer_complements(decoded_val_characters, true_val_characters)


def evaluate_with_rescoring(model, val_loader, phoneme_to_index, index_to_phoneme):
    """
    Evaluate the model with language model rescoring.
    
    Args:
        model (nn.Module): Model to evaluate.
        val_loader (DataLoader): Validation data loader.
        phoneme_to_index (dict): Mapping from phonemes to indices.
        index_to_phoneme (dict): Mapping from indices to phonemes.
    """
    # Initialize language model
    nextsyllable_model = NextSyllableLSTM(
        vocab_size=len(phoneme_to_index),
        embedding_dim=200,
        hidden_dim=512,
        num_layers=4,
        dropout=0.2
    ).to(DEVICE)
    
    # Load model weights
    nextsyllable_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "best_syllable_model_def2.pth"), map_location=DEVICE))
    nextsyllable_model.to(DEVICE)
    nextsyllable_model.eval()
    
    # Decode with rescoring
    blank_token = phoneme_to_index["<UNK>"]
    decoded_val_sequences, true_val_sequences, decoded_val_gestures, true_val_gestures = decode_loader(
        model, val_loader, blank_token, index_to_phoneme, DEVICE, training=False
    )
    
    rescored_sequences, _ = decode_loader_with_rescoring(
        model, nextsyllable_model, val_loader, blank_token, index_to_phoneme, phoneme_to_index, DEVICE
    )
    
    # Calculate PER
    greedy_per = calculate_per_with_jiwer(decoded_val_sequences, true_val_sequences)
    rescored_per = calculate_per_with_jiwer(rescored_sequences, true_val_sequences)
    
    # Calculate PER for phonemes
    phonemes_rescored = [syllables_to_phonemes(seq) for seq in rescored_sequences]
    true_val_characters = [syllables_to_phonemes(seq) for seq in true_val_sequences]
    phoneme_rescored_per = calculate_per_with_jiwer(phonemes_rescored, true_val_characters)
    
    # Calculate PER for gestures
    rescored_gestures = [syllables_to_gestures(seq) for seq in rescored_sequences]
    true_val_gestures = [syllables_to_gestures(seq) for seq in true_val_sequences]
    rescored_gestures_per = calculate_per_with_jiwer(rescored_gestures, true_val_gestures)
    
    # Print results
    print(f"Greedy 1 - PER: {1 - greedy_per:.3f}, Rescored 1 - PER: {1 - rescored_per:.3f}, Rescored phoneme 1 - PER: {1 - phoneme_rescored_per:.3f}")
    print(f"Rescored gestures 1 - PER: {1 - rescored_gestures_per:.3f}")


def main():
    """
    Main function to run the ACSR system.
    """
    # Prepare data
    train_loader, val_loader, phoneme_to_index, index_to_phoneme = prepare_data()
    
    # Check if we should train or load a pre-trained model
    model_path = os.path.join(MODELS_DIR, "model_decoding_last_new_features.pt")
    if os.path.exists(model_path) and "--train" not in sys.argv:
        print(f"Loading pre-trained model from {model_path}")
        # Get sample batch to determine dimensions
        for batch_X_acoustic_hand_shape, batch_X_acoustic_hand_pos, batch_X_acoustic_lips, batch_y in train_loader:
            break
        
        # Initialize model
        model = JointCTCAttentionModel(
            hand_shape_dim=batch_X_acoustic_hand_shape.shape[-1],
            hand_pos_dim=batch_X_acoustic_hand_pos.shape[-1],
            lips_dim=batch_X_acoustic_lips.shape[-1],
            visual_lips_dim=None,
            output_dim=len(phoneme_to_index),
            hidden_dim=ACOUSTIC_MODEL_PARAMS["encoder_hidden_dim"],
        )
        
        # Load model weights
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
    else:
        print("Training new model")
        model = train_acoustic_model(train_loader, val_loader, phoneme_to_index, index_to_phoneme)
    
    # Evaluate model
    if "--evaluate" in sys.argv:
        print("Evaluating model")
        evaluate_model(model, val_loader, phoneme_to_index, index_to_phoneme)
    
    # Evaluate with rescoring
    if "--rescore" in sys.argv:
        print("Evaluating model with rescoring")
        evaluate_with_rescoring(model, val_loader, phoneme_to_index, index_to_phoneme)
    
    # Finish W&B run
    wandb.finish()


if __name__ == "__main__":
    main() 