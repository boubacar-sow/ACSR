# -*- coding: utf-8 -*-
"""
Evaluate Model and Make Predictions

Created on Fri Jun 24 11:36:16 2022

@author: Hagar
"""

import argparse
import logging
import os
import sys
import pandas as pd
import pickle
from utils import (compute_predictions, extract_class_from_fn, load_model,
                   setup_logging,)

__author__ = "Boubacar Sow"
__copyright__ = "Boubacar Sow"
__license__ = "MIT"

# Set up a logger
_logger = logging.getLogger(__name__)


def main(args):
    """Wrapper allowing the script to be called
        with string arguments in a CLI fashion.

    Args:
        args (List[str]): Command line parameters as list of strings
    """
    args = parse_args(args)
    setup_logging(args.loglevel)

    _logger.debug("Starting evaluation...")

    # Load model
    fn_model = os.path.join(
        args.path2models, f'model_{args.model_type}_{args.property_type}.pkl'
    )
    print(load_model(fn_model))
    model, feature_names = load_model(fn_model)
    _logger.info(f'Loaded model: {fn_model}')

    # Find all feature CSV files
    feature_files = [
        f for f in os.listdir(args.path2features)
        if f.endswith('_features.csv')
    ]
    _logger.info(f'Found {len(feature_files)} feature files to process.')

    # Create predictions directory if it doesn't exist
    predictions_dir = os.path.join(args.path2output, 'predictions')
    os.makedirs(predictions_dir, exist_ok=True)

    # Find all frame-to-predict files
    frames_to_predict_files = [
        f for f in os.listdir(args.frames_to_predict)
        if f.endswith('.txt')
    ]
    _logger.info(f'Found {len(frames_to_predict_files)} frame-to-predict files to process.')

    # Process each frame-to-predict file
    for fn_frames_to_predict in frames_to_predict_files:
        # Extract video name from the frame-to-predict file
        video_name = os.path.splitext(fn_frames_to_predict)[0]
        _logger.info(f'Processing video: {video_name}')

        # Find the corresponding features file
        fn_feature = f"{video_name}_features.csv"
        if fn_feature not in feature_files:
            _logger.warning(f"No features file found for video: {video_name}")
            continue

        # Load frames to predict
        frames_to_predict_path = os.path.join(args.frames_to_predict, fn_frames_to_predict)
        with open(frames_to_predict_path, "r") as f:
            frames_to_predict = [int(line.strip()) for line in f.readlines()]
        _logger.info(f"Frames to predict: {frames_to_predict}")

        # Load features
        df_features = pd.read_csv(os.path.join(args.path2features, fn_feature))
        _logger.info(f'Loaded features from: {fn_feature}')

        # Filter frames to predict
        df_features = df_features[df_features['frame_number'].isin(frames_to_predict)]

        # Predict
        predicted_probs, predicted_class = compute_predictions(
            model, df_features[feature_names])

        n_classes = {'position': 5, 'shape': 8}[args.property_type]
        columns = (['frame_number', 'predicted_class'] +
                   [f'p_class_{c + 1}' for c in range(n_classes)])
        df_predictions = pd.DataFrame(columns=columns)

        df_predictions['frame_number'] = df_features['frame_number']
        df_predictions['predicted_class'] = predicted_class

        for i_row, curr_predicted_probs in enumerate(predicted_probs):
            if curr_predicted_probs is not None:
                for c, p_c in enumerate(curr_predicted_probs):
                    df_predictions.loc[i_row, f'p_class_{c + 1}'] = p_c

        # Add zeros for frames not predicted
        all_frames = pd.DataFrame({'frame_number': frames_to_predict})
        df_predictions = pd.merge(all_frames, df_predictions, on='frame_number', how='left').fillna(0)

        # Save predictions
        fn_predictions = os.path.join(
            predictions_dir,
            f'predictions_{args.model_type}_{args.property_type}_'
            f'{video_name}.csv'
        )
        df_predictions.to_csv(fn_predictions, index=False)
        _logger.info(f'Predictions saved to: {fn_predictions}')

    _logger.info("Evaluation complete")


def parse_args(args):
    """Parse command line parameters.

    Args:
        args (List[str]): Command line parameters as list of strings.

    Returns:
        argparse.Namespace: Command line parameters namespace.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate model and make predictions for all videos"
    )
    parser.add_argument(
        '--property-type', choices=['shape', 'position'], default='shape',
        help="Type of property (shape or position)"
    )
    parser.add_argument(
        '--model-type', choices=['rf', 'lr', 'rc', 'gb'], default='rf',
        help="Model type: rf=random-forest, lr=logistic-regression"
    )
    parser.add_argument(
        '--path2models', 
        default=(
            r"/scratch2/bsow/Documents/ACSR/output/saved_models"
        ),
        help="Path to trained models"
    )
    parser.add_argument(
        '--path2features',
        default=(
            r"/scratch2/bsow/Documents/ACSR/output/extracted_features"
        ),
        help="Path to feature CSV files"
    )
    parser.add_argument(
        '--path2output', 
        default=(
            r"/scratch2/bsow/Documents/ACSR/output"
        ),
        help="Path to save output files"
    )
    parser.add_argument(
        '--frames-to-predict', type=str, 
        default=(
            r"/scratch2/bsow/Documents/ACSR/data/training_videos/frames_to_predict"
        ),
        help="Path to the directory containing frame-to-predict files"
    )
    parser.add_argument(
        '--loglevel', default='INFO',
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level"
    )
    return parser.parse_args(args)


def run():
    """Calls :func:`main` passing the CLI arguments extracted from
    :obj:`sys.argv`."""
    main(sys.argv[1:])


if __name__ == "__main__":
    run()