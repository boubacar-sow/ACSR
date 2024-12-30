"""
This script processes video files to extract coordinates and save them to a
CSV file.
"""

import argparse
import glob
import logging
import os
import sys

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import load_video

__author__ = "Boubacar Sow"
__copyright__ = "Boubacar Sow"
__license__ = "MIT"

_logger = logging.getLogger(__name__)


def extract_coordinates(cap, fn_video, show_video=False, verbose=True):

    if verbose:
        print(f"Extracting coordinates for: {fn_video}")
    mp_drawing = mp.solutions.drawing_utils  # Drawing helpers
    mp_holistic = mp.solutions.holistic  # Mediapipe Solutions

    columns = ["fn_video", "frame_number"]
    num_coords_face = 468
    num_coords_hand = 21

    # generate columns names
    for val in range(0, num_coords_face):
        columns += [
            "x_face{}".format(val),
            "y_face{}".format(val),
            "z_face{}".format(val),
            "v_face{}".format(val),
        ]

    for val in range(0, num_coords_hand):
        columns += [
            "x_r_hand{}".format(val),
            "y_r_hand{}".format(val),
            "z_r_hand{}".format(val),
            "v_r_hand{}".format(val),
        ]

    df_coords = pd.DataFrame(columns=columns)

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if verbose:
        print(f"Number of frames in video: {n_frames}")
    pbar = tqdm(total=n_frames)

    # Initiate holistic model
    i_frame = 0
    with mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            i_frame += 1

            if not ret:
                break
            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(
                image
            )

            # Recolor image back to BGR for rendering
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 4. Pose Detections
            if show_video:
                # Draw face landmarks
                mp_drawing.draw_landmarks(
                    image,
                    results.face_landmarks,
                    mp_holistic.FACEMESH_TESSELATION,
                    mp_drawing.DrawingSpec(
                        color=(80, 110, 10), thickness=1, circle_radius=1
                    ),
                    mp_drawing.DrawingSpec(
                        color=(80, 256, 121), thickness=1, circle_radius=1
                    ),
                )

                # Right hand landmarks
                mp_drawing.draw_landmarks(
                    image,
                    results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(
                        color=(80, 22, 10), thickness=2, circle_radius=4
                    ),
                    mp_drawing.DrawingSpec(
                        color=(80, 44, 121), thickness=2, circle_radius=2
                    ),
                )
                # Pose landmarks
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(
                        color=(245, 117, 66), thickness=2, circle_radius=4
                    ),
                    mp_drawing.DrawingSpec(
                        color=(245, 66, 230), thickness=2, circle_radius=2
                    ),
                )
                cv2.imshow("cued_estimated", image)

            # Export coordinates
            if results.face_landmarks is not None:
                face = results.face_landmarks.landmark
                face_row = list(
                    np.array(
                        [
                            [
                                landmark.x, landmark.y, landmark.z,
                                landmark.visibility
                            ]
                            for landmark in face
                        ]
                    ).flatten()
                )

            else:
                face_row = [None] * 4
            # Extract right hand landmarks
            if results.right_hand_landmarks is not None:
                r_hand = results.right_hand_landmarks.landmark
                r_hand_row = list(
                    np.array(
                        [
                            [
                                landmark.x, landmark.y, landmark.z,
                                landmark.visibility
                            ]
                            for landmark in r_hand
                        ]
                    ).flatten()
                )
            else:
                r_hand_row = [None] * 4

            # Create the row that will be written in the file
            row = [fn_video, i_frame] + face_row + r_hand_row
            curr_df = pd.DataFrame(dict(zip(columns, row)), index=[0])
            # print(i_frame, curr_df)
            df_coords = pd.concat([df_coords, curr_df], ignore_index=True)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
                print("WARNING!" * 5)
                print('break due to cv2.waitKey(10) & 0xFF == ord("q"')
            pbar.update(1)

    cap.release()
    cv2.destroyAllWindows()

    # print(len(df_coords), n_frames)
    assert n_frames - df_coords.shape[0] <= 1

    return df_coords


def process_videos(show_video, gender, cropping, path2data, path2output):
    """Process video files and extract coordinates

    Args:
        show_video (bool): Whether to display the video during processing
        gender (str): Gender of the videos to process (male or female)
        cropping (str): Cropping method (cropped or non_cropped)
        path2data (str): Path to the directory containing the video files
        path2output (str): Path to the directory to save the output CSV
    """
    path_pattern = os.path.join(path2data, '*.mp4')
    fn_videos = glob.glob(path_pattern)

    df = pd.DataFrame()

    for fn_video in fn_videos[:20]:
        _logger.info(f'Loading: {fn_video}')
        cap = load_video(fn_video)
        df_coords = extract_coordinates(
            cap, os.path.basename(fn_video), show_video=show_video,
            verbose=True
        )
        df = pd.concat([df, df_coords])

    os.makedirs(path2output, exist_ok=True)

    fn_output = f'training_coords_face_hand_{gender}_{cropping}.csv'
    fn_output = os.path.join(path2output, fn_output)
    df.to_csv(fn_output)
    _logger.info(f'Coordinates saved to: {fn_output}')


def parse_args(args):
    """Parse command line parameters

    Args:
        args (List[str]): Command line parameters as list of strings

    Returns:
        argparse.Namespace: Command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="Process video files to extract coordinates"
    )
    parser.add_argument(
        '--show-video', action='store_true', default=False,
        help="Show video during processing"
    )
    parser.add_argument(
        '--gender', default='female', choices=['male', 'female'],
        help="Gender of videos to process"
    )
    parser.add_argument(
        '--cropping', default='cropped', choices=['cropped', 'non_cropped'],
        help="Cropping method"
    )
    parser.add_argument(
        '--path2data', default=os.path.join('ACSR', 'data', 'training_videos'),
        help="Path to video files"
    )
    parser.add_argument(
        '--path2output', default=os.path.join('ACSR', 'output'),
        help="Path to save output CSV"
    )
    parser.add_argument(
        '--loglevel', default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level"
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
    _logger.debug("Starting video processing...")
    process_videos(
        args.show_video, args.gender, args.cropping, args.path2data,
        args.path2output
    )
    _logger.info("Script ends here")


def run():
    """Calls :func:`main` passing the CLI arguments extracted from
    :obj:`sys.argv`. This function can be used as entry point to create
    console scripts with setuptools. """

    main(sys.argv[1:])


if __name__ == "__main__":
    run()
