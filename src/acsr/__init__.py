"""
ACSR (Automatic Cued Speech Recognition) system.
This package contains modules for training and evaluating models for cued speech recognition.
"""

from .models import JointCTCAttentionModel, NextSyllableLSTM
from .data import load_csv_files, find_phoneme_files, prepare_data_for_videos_no_sliding_windows, data_to_dataloader
from .utils.text_processing import syllabify_ipa, syllables_to_gestures, syllables_to_phonemes
from .utils.metrics import calculate_per_with_jiwer, compute_edit_distance, compute_and_print_wer_complements
from .utils.decoding_utils import greedy_decoder, logsumexp, compute_lm_score
from .train import train_model, validate_model, joint_ctc_attention_loss
from .decode import decode_loader, decode_loader_with_rescoring, rescore_sequences
from .main import main, prepare_data, evaluate_model, evaluate_with_rescoring

__version__ = "1.0.0"

__all__ = [
    'JointCTCAttentionModel',
    'NextSyllableLSTM',
    'load_csv_files',
    'find_phoneme_files',
    'prepare_data_for_videos_no_sliding_windows',
    'data_to_dataloader',
    'syllabify_ipa',
    'syllables_to_gestures',
    'syllables_to_phonemes',
    'calculate_per_with_jiwer',
    'compute_edit_distance',
    'compute_and_print_wer_complements',
    'greedy_decoder',
    'logsumexp',
    'compute_lm_score',
    'train_model',
    'validate_model',
    'joint_ctc_attention_loss',
    'decode_loader',
    'decode_loader_with_rescoring',
    'rescore_sequences',
    'main',
    'prepare_data',
    'evaluate_model',
    'evaluate_with_rescoring',
]
