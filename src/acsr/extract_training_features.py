# -*- coding: utf-8 -*-
"""
This script processes coordinate CSV files to extract features and save them
to a CSV file.

Created on Wed Jun 22 11:23:06 2022

@author: hagar
"""

import argparse
import logging
import os
import sys
import pandas as pd
from utils import get_delta_dim, get_distance, get_index_pairs

__author__ = "Boubacar Sow"
__copyright__ = "Boubacar Sow"
__license__ = "MIT"

_logger = logging.getLogger(__name__)


def extract_features(df_coords):
    # create the df of relevant feature

    df_features = pd.DataFrame()
    df_features["fn_video"] = df_coords["fn_video"].copy()
    df_features["frame_number"] = df_coords["frame_number"]

    # face width to normalize the distance
    # print('Computing face width for normalization')
    face_width = get_distance(df_coords, "face234", "face454").mean()
    norm_factor = face_width
    print(f"Face width computed for normalizaiton {face_width}")

    # norm_factor = None # REMOVE NORMALIZAION

    # HAND-FACE DISTANCES AS FEATURES FOR POSITION DECODING
    position_index_pairs = get_index_pairs("position")
    for hand_index, face_index in position_index_pairs:
        feature_name = f"distance_face{face_index}_r_hand{hand_index}"
        # print(f'Computing {feature_name}')
        df_features[feature_name] = get_distance(
            df_coords,
            f"face{face_index}",
            f"r_hand{hand_index}",
            norm_factor=norm_factor,
        )

        dx = get_delta_dim(
            df_coords,
            f"face{face_index}",
            f"r_hand{hand_index}",
            "x",
            norm_factor=norm_factor,
        )

        dy = get_delta_dim(
            df_coords,
            f"face{face_index}",
            f"r_hand{hand_index}",
            "y",
            norm_factor=norm_factor,
        )

        feature_name = f"tan_angle_face{face_index}_r_hand{hand_index}"
        df_features[feature_name] = dx / dy

    # HAND-HAND DISTANCES AS FEATURE FOR SHAPE DECODING
    shape_index_pairs = get_index_pairs("shape")
    for hand_index1, hand_index2 in shape_index_pairs:
        feature_name = f"distance_r_hand{hand_index1}_r_hand{hand_index2}"
        # print(f'Computing {feature_name}')
        df_features[feature_name] = get_distance(
            df_coords,
            f"r_hand{hand_index1}",
            f"r_hand{hand_index2}",
            norm_factor=norm_factor,
        )

    return df_features


def process_coordinates(gender, cropping, path2coordinates):
    """Process coordinate CSV files and extract features

    Args:
        gender (str): Gender of the data (male or female)
        cropping (str): Cropping method (cropped or non_cropped)
        path2coordinates (str): Path to the directory containing the \
                                coordinate CSV files
    """
    fn_coordinates = f'training_coords_face_hand_{gender}_{cropping}.csv'
    df_coord = pd.read_csv(os.path.join(path2coordinates, fn_coordinates))
    _logger.info(f'Loaded coordinates from: {fn_coordinates}')
    _logger.debug(f'Coordinate files: {set(df_coord["fn_video"])}')

    # Extract features
    df_features = extract_features(df_coord)

    # Save features to CSV
    fn_features = f'training_features_{gender}_{cropping}.csv'
    fn_features_path = os.path.join(path2coordinates, fn_features)
    df_features.to_csv(fn_features_path)
    _logger.info(f'Features saved to: {fn_features_path}')


def parse_args(args):
    """Parse command line parameters

    Args:
        args (List[str]): Command line parameters as list of strings

    Returns:
        argparse.Namespace: Command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="Process coordinate CSV files to extract features"
    )
    parser.add_argument(
        '--gender', default='female', choices=['male', 'female'],
        help="Gender of the data"
    )
    parser.add_argument(
        '--cropping', default='cropped', choices=['cropped', 'non_cropped'],
        help="Cropping method"
    )
    parser.add_argument(
        '--path2coordinates', default=os.path.join('ACSR', 'output'),
        help="Path to coordinate CSV files"
    )
    parser.add_argument(
        '--loglevel', default='INFO',
        help="Logging level (one of: DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
        loglevel (int): Minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat,
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def main(args):
    """Wrapper allowing the script to be called with string arguments in a CLI
    fashion

    Args:
        args (List[str]): Command line parameters as list of strings
    """
    args = parse_args(args)
    setup_logging(args.loglevel)
    _logger.debug("Starting coordinate processing...")
    process_coordinates(args.gender, args.cropping, args.path2coordinates)
    _logger.info("Script ends here")


def run():
    """Calls :func:`main` passing the CLI arguments extracted from
    :obj:`sys.argv`. This function can be used as entry point to create
    console scripts with setuptools.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
