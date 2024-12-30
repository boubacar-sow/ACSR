# -*- coding: utf-8 -*-
"""
Visualization and Analysis Script

Created on Mon Jun 27 14:08:44 2022

"""

import argparse
import os
import pandas as pd
from pathlib import Path
import PyQt5
import sys
import utils
import viz


def parse_args(args):
    """Parse command line parameters

    Args:
        args (List[str]): Command line parameters as list of strings

    Returns:
        argparse.Namespace: Command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="Visualize and analyze video predictions and measures"
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
        '--model-type', choices=['rf', 'lr', 'rc', 'gb'], default='lr',
        help='Model type: rf=random-forest, lr=logistic-regression'
    )
    parser.add_argument(
        '--fn-video', default='test.mp4', help="Filename of the video"
    )
    parser.add_argument(
        '--path2video', default=os.path.join('ACSR', 'data', 'test_videos'),
        help="Path to video files"
    )
    parser.add_argument(
        '--path2predictions', default=os.path.join('ACSR', 'output'),
        help="Path to prediction files"
    )
    parser.add_argument(
        '--path2output', default=os.path.join('ACSR', 'output'),
        help="Path to save output files"
    )
    parser.add_argument(
        '--path2figures', default=os.path.join('ACSR', 'figures'),
        help="Path to save figures"
    )
    parser.add_argument(
        '--textgrid', action='store_true', default=True,
        help='If true, onset from grid text will be added'
    )
    parser.add_argument(
        '--plot-measures', action='store_true', default=False,
        help=('If true, velocity, joint measures, and probabilities will be '
              'plotted')
    )
    parser.add_argument(
        '--weight-velocity', default=3, type=float,
        help=('Importance weight of velocity compared to probabilities '
              'in the computation of the joint measure')
    )
    return parser.parse_args(args)


def main(args):
    """Main function for visualizing and analyzing video predictions

    Args:
        args (argparse.Namespace): Parsed command line arguments
    """
    args = parse_args(args)

    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.fspath(
        Path(PyQt5.__file__).resolve().parent / "Qt5" / "plugins"
    )

    # Load video
    fn_video = os.path.join(args.path2video, args.fn_video)
    cap = utils.load_video(fn_video)
    print(f'Visualization for: {fn_video}')
    print(cap.__sizeof__())

    # Load predictions
    fn_predictions_pos = (f'predictions_{args.model_type}_position_'
                          f'{args.gender}_{args.cropping}_'
                          f'{args.fn_video[:-4]}.csv')
    df_predictions_pos = pd.read_csv(
        os.path.join(args.path2predictions, fn_predictions_pos)
    )

    fn_predictions_shape = (f'predictions_{args.model_type}_shape_'
                            f'{args.gender}_{args.cropping}_'
                            f'{args.fn_video[:-4]}.csv')
    df_predictions_shape = pd.read_csv(
        os.path.join(args.path2predictions, fn_predictions_shape)
    )

    # Load coordinate DataFrame
    df_coord = pd.read_csv(os.path.join(
        args.path2output, f'{args.fn_video[:-4]}_coordinates.csv'))

    # Compute velocity and scale it
    velocity, acceleration = utils.compute_velocity(
        df_coord, 'r_hand9',
        fn=f'../output/velocity_{args.fn_video}'
    )
    velocity_scaled = utils.scale_velocity(velocity)

    # Get stimulus string
    str_stimulus = utils.get_stimulus_string(fn_video)

    # Get syllable onsets
    lpc_syllables = utils.get_syllable_onset_frames_from_lpc_file(fn_video)
    n_syllables = len(lpc_syllables)
    print(lpc_syllables)

    if args.textgrid:
        onset_frames_syllables_mfa = utils.get_syllable_onset_frames_from_mfa(
            fn_video, lpc_syllables)
        print(f'Number of MFA syllables found: {n_syllables}')
    else:
        lpc_syllables = None
        onset_frames_syllables_mfa = None

    # Compute joint measure
    joint_measure = utils.get_joint_measure(
        df_predictions_pos, df_predictions_shape, velocity_scaled,
        weight_velocity=args.weight_velocity)

    print(f'Stimulus: {str_stimulus}')
    if (lpc_syllables is not None) and \
       (onset_frames_syllables_mfa is not None):
        [print(f'{syl}: frame #{onset}')
         for syl, onset in zip(lpc_syllables, onset_frames_syllables_mfa)]

    # Get onset frames based on joint measure
    thresh = 0.3
    onset_frames_picked, onset_frames_extrema = \
        utils.find_onsets_based_on_extrema(
            joint_measure, n_syllables,
            onset_frames_syllables_mfa=onset_frames_syllables_mfa,
            thresh=thresh)

    print('Frame onsets of extrema of joint measure:', onset_frames_extrema)
    print('Identified frame onsets:', onset_frames_picked)

    # Plot measures
    if args.plot_measures:
        fig, _ = viz.plot_joint_measure(
            df_predictions_pos, df_predictions_shape, velocity_scaled,
            joint_measure, lpc_syllables, onset_frames_syllables_mfa,
            onsets_extrema=onset_frames_picked)

        os.makedirs(args.path2figures, exist_ok=True)
        fn_fig = os.path.join(
            args.path2video, f'{os.path.basename(fn_video)}.png')
        fig.savefig(fn_fig)
        print(f'Figure was saved to: {fn_fig}')

    os.makedirs(args.path2output, exist_ok=True)
    fn_txt = os.path.join(
        args.path2video, f'{os.path.basename(fn_video)}.events')
    utils.write_onsets_to_file(
        str_stimulus, lpc_syllables, onset_frames_picked, fn_txt)
    print(f'Event onsets saved to: {fn_txt}')

    # Save measures to CSV
    df_measures = pd.DataFrame(list(zip(
        velocity, acceleration, velocity_scaled, joint_measure)),
        columns=['velocity', 'acceleration', 'velocity_scaled',
                 'joint_measure'])
    df_measures.to_csv(os.path.join(
        args.path2output, f'{args.fn_video[:-4]}_measures.csv'))


def run():
    """Calls :func:`main` passing the CLI arguments extracted from
    :obj:`sys.argv`. This function can be used as entry point to create
    console scripts with setuptools.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
