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

    for fn_feature in feature_files:
        # Load features
        df_features = pd.read_csv(os.path.join(args.path2features, fn_feature))
        _logger.info(f'Loaded features from: {fn_feature}')

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
        # Extract video name
        video_name = fn_feature.replace('_features.csv', '')
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
        '--property-type', choices=['shape', 'position'], default='position',
        help="Type of property (shape or position)"
    )
    parser.add_argument(
        '--model-type', choices=['rf', 'lr', 'rc', 'gb'], default='rf',
        help="Model type: rf=random-forest, lr=logistic-regression"
    )
    parser.add_argument(
        '--path2models', default=os.path.join('ACSR', 'trained_models'),
        help="Path to trained models"
    )
    parser.add_argument(
        '--path2features',
        default=os.path.join('ACSR', 'output', 'extracted_features'),
        help="Path to feature CSV files"
    )
    parser.add_argument(
        '--path2output', default=os.path.join('ACSR', 'output'),
        help="Path to save output files"
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
