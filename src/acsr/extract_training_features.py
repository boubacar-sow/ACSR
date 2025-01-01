# -*- coding: utf-8 -*-
"""
This script processes coordinate CSV files to extract features and save them
to individual CSV files.
"""

import argparse
import logging
import os
import sys
import pandas as pd
from utils import extract_features

__author__ = "Boubacar Sow"
__copyright__ = "Boubacar Sow"
__license__ = "MIT"

_logger = logging.getLogger(__name__)


def process_coordinates(path2coordinates, path2output):
    """Process coordinate CSV files and extract features.

    Args:
        path2coordinates (str): Path to the directory containing the coordinate
                                CSV files.
        path2output (str): Path to the directory to save the feature CSV files.
    """
    # Find all coordinate CSV files
    coordinate_files = [
        f for f in os.listdir(path2coordinates)
        if f.endswith('_coordinates.csv')
    ]
    _logger.info(f'Found {len(coordinate_files)} coordinate files to process.')

    # Create output directory if it doesn't exist
    os.makedirs(path2output, exist_ok=True)

    for fn_coord in coordinate_files:
        # Load coordinates
        df_coord = pd.read_csv(os.path.join(path2coordinates, fn_coord))
        _logger.info(f'Loaded coordinates from: {fn_coord}')

        # Extract features
        df_features = extract_features(df_coord)

        # Save features to CSV
        # Extract video name
        video_name = fn_coord.replace('_coordinates.csv', '')
        fn_features = f'{video_name}_features.csv'
        fn_features_path = os.path.join(path2output, fn_features)
        df_features.to_csv(fn_features_path, index=False)
        _logger.info(f'Features saved to: {fn_features_path}')


def parse_args(args):
    """Parse command line parameters.

    Args:
        args (List[str]): Command line parameters as list of strings.

    Returns:
        argparse.Namespace: Command line parameters namespace.
    """
    parser = argparse.ArgumentParser(
        description="Process coordinate CSV files to extract features"
    )
    parser.add_argument(
        '--path2coordinates',
        default=os.path.join('ACSR', 'output', 'extracted_coordinates'),
        help="Path to coordinate CSV files"
    )
    parser.add_argument(
        '--path2output',
        default=os.path.join('ACSR', 'output', 'extracted_features'),
        help="Path to save feature CSV files"
    )
    parser.add_argument(
        '--loglevel',
        default='INFO',
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level"
    )
    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging.

    Args:
        loglevel (int): Minimum loglevel for emitting messages.
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat,
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def main(args):
    """Wrapper allowing the script to be called with string arguments in a CLI
    fashion.

    Args:
        args (List[str]): Command line parameters as list of strings.
    """
    args = parse_args(args)
    setup_logging(args.loglevel)
    _logger.debug("Starting coordinate processing...")
    process_coordinates(args.path2coordinates, args.path2output)
    _logger.info("Script ends here")


def run():
    """Calls :func:`main` passing the CLI arguments extracted from
    :obj:`sys.argv`."""
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
