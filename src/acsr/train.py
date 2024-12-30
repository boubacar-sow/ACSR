# -*- coding: utf-8 -*-
"""
Train Model Using Scikit Learn

Created on Wed May 18 15:06:34 2022

@author: Hagar
"""

import argparse
import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from utils import setup_logging

__author__ = "Boubacar Sow"
__copyright__ = "Boubacar Sow"
__license__ = "MIT"

# Set up a logger
_logger = logging.getLogger(__name__)


def get_index_pairs(property_type):
    index_pairs = []
    if property_type == 'shape':
        index_pairs.extend([
            (2, 4), (5, 8), (9, 12), (13, 16), (17, 20),
            (4, 5), (4, 8), (8, 12), (7, 11), (6, 10)
        ])
    elif property_type == 'position':
        hand_indices = [8, 9, 12]  # index and middle fingers
        face_indices = [130, 152, 94]  # right eye, chin, nose
        for hand_index in hand_indices:
            for face_index in face_indices:
                index_pairs.append((hand_index, face_index))
    return index_pairs


def get_feature_names(property_name):
    feature_names = []
    if property_name == 'position':
        position_index_pairs = get_index_pairs('position')
        for hand_index, face_index in position_index_pairs:
            feature_name = (f'distance_face{face_index}_r_hand{hand_index}')
            feature_names.append(feature_name)
            feature_name = (f'tan_angle_face{face_index}_r_hand{hand_index}')
            feature_names.append(feature_name)
    elif property_name == 'shape':
        shape_index_pairs = get_index_pairs('shape')
        for hand_index1, hand_index2 in shape_index_pairs:
            feature_name = (
                f'distance_r_hand{hand_index1}_r_hand{hand_index2}'
            )
            feature_names.append(feature_name)
    return feature_names


def train_model():
    args = parse_args(sys.argv[1:])
    setup_logging(args.loglevel)

    _logger.debug("Starting model training...")

    df_features = pd.read_csv(
        os.path.join(args.path2features,
                     f'training_features_{args.gender}_{args.cropping}.csv'),
        index_col=0
    )
    feature_names = get_feature_names(args.property_type)

    df_features = df_features[['fn_video'] + feature_names]
    # df_features = df_features.loc[
    #     df_features['fn_video'].str.contains(args.property_type, regex=False)
    # ]

    df_features = df_features.dropna()
    print(len(df_features))
    if args.verbose:
        print(df_features)
        print('Training with the following features:')
        print(feature_names)

    y = df_features['fn_video']
    X = df_features.drop(['fn_video'], axis=1)

    p_grid = {"C": [10**i for i in range(-5, 6)]}
    pipelines = {
        'sv': LinearSVC(),
        'lr': LogisticRegression(max_iter=1000000),
        'rc': make_pipeline(RidgeClassifier()),
        'rf': make_pipeline(RandomForestClassifier()),
        'gb': make_pipeline(GradientBoostingClassifier())
    }
    pipeline = pipelines[args.model_type]

    k_in, k_out = 2, 2
    inner_cv = KFold(n_splits=k_in, shuffle=True, random_state=1234)
    outer_cv = KFold(n_splits=k_out, shuffle=True, random_state=1234)

    clf = GridSearchCV(estimator=pipeline, param_grid=p_grid, cv=inner_cv)
    clf.fit(X, y)

    nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv)
    print(f'Nested cross-validated score for {args.property_type}: '
          f'{np.mean(nested_score)} +- {np.std(nested_score)}')

    model = pipeline.fit(X, y)
    os.makedirs(os.path.join('ACSR', 'trained_models'), exist_ok=True)
    file_name = os.path.join(
        'ACSR', 'trained_models',
        f'model_{args.model_type}_{args.property_type}_'
        f'{args.gender}_{args.cropping}.pkl'
    )
    with open(file_name, 'wb') as f:
        pickle.dump([model, feature_names], f)

    _logger.info(f'Model and feature names saved to {file_name}')


def parse_args(args):
    """Parse command line parameters

    Args:
        args (List[str]): Command line parameters as list of strings

    Returns:
        argparse.Namespace: Command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="Train a model using Scikit Learn"
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
        '--path2features', default=os.path.join('ACSR', 'output'),
        help="Path to feature CSV files"
    )
    parser.add_argument(
        '--property-type', choices=['shape', 'position'], default='position',
        help="Type of property (shape or position)"
    )
    parser.add_argument(
        '--model-type', choices=['sv', 'rf', 'lr', 'rc', 'gb'], default='lr',
        help="Model type: rf=random-forest, lr=logistic-regression"
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true', default=False,
        help="Enable verbose output"
    )
    parser.add_argument(
        '--loglevel', default='INFO',
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    return parser.parse_args(args)


def main(args):
    """Wrapper allowing the script to be called with string arguments in a CLI
    fashion

    Args:
        args (List[str]): Command line parameters as list of strings
    """
    args = parse_args(args)
    setup_logging(args.loglevel)
    _logger.info("Starting model training...")
    train_model()
    _logger.info("Script ends here")


def run():
    """Calls :func:`main` passing the CLI arguments extracted from
    :obj:`sys.argv`. This function can be used as entry point to create
    console scripts with setuptools.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
