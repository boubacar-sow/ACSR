"""
Configuration settings for the ACSR (Automatic Cued Speech Recognition) system.
This file centralizes all configuration parameters used across the system.
"""

import os
import torch

# Paths
DATA_DIR = "/pasteur/appa/homes/bsow/ACSR/data"
OUTPUT_DIR = "/pasteur/appa/homes/bsow/ACSR/output"
MODELS_DIR = "/pasteur/appa/homes/bsow/ACSR/src/acsr/saved_models"

# Training videos paths
TRAIN_VIDEOS_DIR = os.path.join(DATA_DIR, "training_videos/CSF22_train")
TEST_VIDEOS_DIR = os.path.join(DATA_DIR, "training_videos/CSF22_test")
TRAIN_LABELS_DIR = os.path.join(TRAIN_VIDEOS_DIR, "train_labels")
TEST_LABELS_DIR = os.path.join(TEST_VIDEOS_DIR, "test_labels")
LIPS_DIR = os.path.join(TRAIN_VIDEOS_DIR, "lip_rois_mp4")

# Features paths
FEATURES_DIR_TRAIN = os.path.join(OUTPUT_DIR, "extracted_features_train_annahita")
FEATURES_DIR_TEST = os.path.join(OUTPUT_DIR, "extracted_features_test_annahita")
COORDINATES_DIR_TRAIN = os.path.join(OUTPUT_DIR, "extracted_coordinates_train")
COORDINATES_DIR_TEST = os.path.join(OUTPUT_DIR, "extracted_coordinates_test")

# French dataset paths
FRENCH_DATASET_DIR = os.path.join(DATA_DIR, "french_dataset")
VOCAB_FILE = os.path.join(FRENCH_DATASET_DIR, "vocab.txt")
PREPROCESSED_TRAIN_PATH = os.path.join(FRENCH_DATASET_DIR, "preprocessed_train.txt")
PREPROCESSED_VAL_PATH = os.path.join(FRENCH_DATASET_DIR, "preprocessed_val.txt")
DATA_TRAIN_PATH = os.path.join(FRENCH_DATASET_DIR, "concat_train.csv")
DATA_VAL_PATH = os.path.join(FRENCH_DATASET_DIR, "eval.csv")

# Model parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Acoustic model parameters
ACOUSTIC_MODEL_PARAMS = {
    "learning_rate": 1e-3,
    "batch_size": 16,
    "epochs": 8000,
    "encoder_hidden_dim": 128,
    "n_layers_gru": 2,
    "alpha": 0.4,  # Weight for CTC loss
    "device": DEVICE,
    "level": "syllables",
}

# Language model parameters
LM_PARAMS = {
    "seq_length": 15,
    "batch_size": 2048,
    "embedding_dim": 200,
    "hidden_dim": 512,
    "num_layers": 4,
    "dropout": 0.1,
    "learning_rate": 0.001,
    "epochs": 200,
    "device": DEVICE,
    "min_count": 1,
    "special_tokens": ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]
}

# Decoding parameters
DECODING_PARAMS = {
    "beam_width": 15,
    "alpha": 0.5,  # Weight for acoustic model in joint decoding
    "threshold": 0.7,  # Confidence threshold for considering alternative tokens
    "top_k": 3,  # Number of top alternatives to consider
}

# Saved model paths
ACOUSTIC_MODEL_PATH = os.path.join(MODELS_DIR, "model_decoding_last_new_features.pt")
SYLLABLE_MODEL_PATH = os.path.join(MODELS_DIR, "best_syllable_model_def2.pth") 