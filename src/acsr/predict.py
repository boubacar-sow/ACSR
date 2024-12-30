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
from extract_training_coordinates import extract_coordinates
from extract_training_features import extract_features
from utils import (compute_predictions, extract_class_from_fn, load_model,
                   load_video, setup_logging)

__author__ = "Boubacar Sow"
__copyright__ = "Boubacar Sow"
__license__ = "MIT"

# Set up a logger
_logger = logging.getLogger(__name__)


def main(args):
    """Wrapper allowing the script to be called with string arguments in a CLI
    fashion

    Args:
        args (List[str]): Command line parameters as list of strings
    """
    args = parse_args(args)
    setup_logging(args.loglevel)

    _logger.debug("Starting evaluation...")

    # Load model
    fn_model = (f'model_{args.model_type}_{args.property_type}_{args.gender}_'
                f'{args.cropping}.pkl')
    fn_model = os.path.join(args.path2models, fn_model)
    model, feature_names = load_model(fn_model)
    print(f'Loaded model: {fn_model}')

    # Load video
    fn_video = os.path.join(args.path2test_videos, args.fn_video)
    if not os.path.isfile(fn_video):
        _logger.error(f'Video file not found: {fn_video}')
        raise FileNotFoundError('Video not found')
    cap = load_video(fn_video)
    print(f'Loaded video: {fn_video}')

    # Extract coordinates
    print('Extracting coordinates...')
    df_coords = extract_coordinates(cap, args.fn_video)

    # Extract features
    print('Extracting features...')
    df_features = extract_features(df_coords)
    if args.save_features:
        df_coords.to_csv(os.path.join(
            args.path2output, f'{args.fn_video[:-4]}_coordinates.csv'))
        df_features.to_csv(os.path.join(
            args.path2output, f'{args.fn_video[:-4]}_features.csv'))
        print(f"Features saved to: ",
              f"{os.path.join(args.path2output, f'{args.fn_video[:-4]}_features.csv')}")

    # Predict
    predicted_probs, predicted_class = compute_predictions(
        model, df_features[feature_names])

    n_classes = {'position': 5, 'shape': 8}[args.property_type]
    columns = (['frame_number', 'predicted_class'] +
               [f'p_class_{c + 1}' for c in range(n_classes)])
    df_predictions = pd.DataFrame(columns=columns)

    df_predictions['frame_number'] = df_features['frame_number']
    df_predictions['predicted_class'] = predicted_class
    df_predictions['predicted_class'] = df_predictions.apply(
        lambda row: extract_class_from_fn(row['predicted_class']), axis=1)

    for i_row, curr_predicted_probs in enumerate(predicted_probs):
        if curr_predicted_probs is not None:
            for c, p_c in enumerate(curr_predicted_probs):
                df_predictions.loc[i_row, f'p_class_{c + 1}'] = p_c

    # Save predictions
    fn_predictions = (f'predictions_{args.model_type}_{args.property_type}_'
                      f'{args.gender}_{args.cropping}_'
                      f'{args.fn_video[:-4]}.csv')
    fn_predictions = os.path.join(args.path2output, fn_predictions)
    df_predictions.to_csv(fn_predictions)
    print(f'CSV file with predictions was saved to: {fn_predictions}')

    _logger.info("Evaluation complete")


def parse_args(args):
    """Parse command line parameters

    Args:
        args (List[str]): Command line parameters as list of strings

    Returns:
        argparse.Namespace: Command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="Evaluate model and make predictions using test videos"
    )
    parser.add_argument(
        '--gender', default='female', choices=['male', 'female'],
        help="Gender of the data"
    )
    parser.add_argument(
        '--cropping', default='cropped',
        choices=['cropped', 'non_cropped'],
        help="Cropping method"
    )
    parser.add_argument(
        '--property-type', choices=['shape', 'position'], default='position',
        help="Type of property (shape or position)"
    )
    parser.add_argument(
        '--model-type', choices=['rf', 'lr', 'rc', 'gb'], default='lr',
        help="Model type: rf=random-forest, lr=logistic-regression"
    )
    parser.add_argument(
        '--fn-video', default='test.mp4', help="Filename of the video"
    )
    parser.add_argument(
        '--path2models', default=os.path.join('ACSR', 'trained_models'),
        help="Path to trained models"
    )
    parser.add_argument(
        '--path2test-videos',
        default=os.path.join('ACSR', 'data', 'test_videos'),
        help="Path to test videos"
    )
    parser.add_argument(
        '--path2output', default=os.path.join('ACSR', 'output'),
        help="Path to save output files"
    )
    parser.add_argument(
        '--save-features', action='store_true', default=True,
        help="Flag to save extracted features"
    )
    parser.add_argument(
        '--loglevel', default='INFO',
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    return parser.parse_args(args)


def run():
    """Calls :func:`main` passing the CLI arguments extracted from
    :obj:`sys.argv`. This function can be used as entry point to create
    console scripts with setuptools.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
