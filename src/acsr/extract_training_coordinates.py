"""
This script processes video files to extract coordinates and save them to
individual CSV files. It skips videos whose coordinates have already been
extracted.
"""

import argparse
import glob
import logging
import os
import sys
from utils import load_video, extract_coordinates

__author__ = "Boubacar Sow and Hagar"
__copyright__ = "Boubacar Sow"
__license__ = "MIT"

_logger = logging.getLogger(__name__)


def process_videos(show_video, path2data, path2output, num_videos):
    """Process video files and extract coordinates.

    Args:
        show_video (bool): Whether to display the video during processing.
        path2data (str): Path to the directory containing the video files.
        path2output (str): Path to the directory to save the output CSV files.
        num_videos (int): Number of videos to process.
    """
    # Find all video paths
    video_paths = glob.glob(os.path.join(path2data, "*", "*.webm"))
    video_paths.sort()  # Sort alphabetically

    # Limit the number of videos to process
    if num_videos > 0:
        video_paths = video_paths[:num_videos]

    # Create output directory if it doesn't exist
    os.makedirs(path2output, exist_ok=True)

    for fn_video in video_paths:
        # Extract video name without extension
        video_name = os.path.basename(fn_video).split('.')[0]
        fn_output = os.path.join(path2output, f'{video_name}_coordinates.csv')

        # Skip processing if the coordinates file already exists
        if os.path.exists(fn_output):
            _logger.info(f'Coordinates already extracted for: {fn_video}')
            continue

        _logger.info(f'Processing: {fn_video}')
        cap = load_video(fn_video)
        df_coords = extract_coordinates(
            cap,
            os.path.basename(fn_video),
            show_video=show_video,
            verbose=True
        )

        # Save coordinates to a separate CSV file for each video
        df_coords.to_csv(fn_output, index=False)
        _logger.info(f'Coordinates saved to: {fn_output}')


def parse_args(args):
    """Parse command line parameters.

    Args:
        args (List[str]): Command line parameters as list of strings.

    Returns:
        argparse.Namespace: Command line parameters namespace.
    """
    parser = argparse.ArgumentParser(
        description="Process video files to extract coordinates"
    )
    parser.add_argument(
        '--show-video', action='store_true', default=False,
        help="Show video during processing"
    )
    parser.add_argument(
        '--path2data',
        default=(
            r'C:\Users\bouba\Downloads\CSF22\CSF22\CSF22_train\train_videos'
        ),
        help="Path to video files"
    )
    parser.add_argument(
        '--path2output',
        default=os.path.join('ACSR', 'output', 'extracted_coordinates'),
        help="Path to save output CSV files"
    )
    parser.add_argument(
        '--num-videos', type=int, default=300,
        help="Number of videos to process (default: all)"
    )
    parser.add_argument(
        '--loglevel', default="INFO",
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
    _logger.debug("Starting video processing...")
    process_videos(
        args.show_video, args.path2data, args.path2output, args.num_videos
    )
    _logger.info("Script ends here")


def run():
    """
    Calls :func:`main` passing the CLI arguments extracted from
    :obj:`sys.argv`.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
