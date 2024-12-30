# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 14:08:44 2022

"""
import argparse
import os
import pandas as pd
from pathlib import Path
import PyQt5
import utils
import viz

parser = argparse.ArgumentParser()
parser.add_argument('--gender', default='female', choices=['male', 'female'])
parser.add_argument('--cropping', default='cropped', choices=['cropped', 'non_cropped'])
parser.add_argument('--model-type', choices=['rf', 'lr', 'rc', 'gb'],
                    help = 'rf:random-forest; lr:logisitic-regrssion',
                    default='rf')
parser.add_argument('--fn-video', default='word_h0_10.mp4')
parser.add_argument('--path2video', default=os.path.join('..', 'stimuli',
                                                         'words', 'mp4'))
parser.add_argument('--path2predictions', default=os.path.join('..',
                                                               'output'))
parser.add_argument('--path2output', default=os.path.join('..', 'output'))
parser.add_argument('--text-factor', default=1, type=float)
parser.add_argument('--textgrid', action='store_true', default=True,
                    help='If true, onset from grid text will be added')
parser.add_argument('--show-video', action='store_true', default=False)
args = parser.parse_args()

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.fspath(
    Path(PyQt5.__file__).resolve().parent / "Qt5" / "plugins"
)


# LOAD VIDEO
fn_video = os.path.join(args.path2video, args.fn_video)
cap = utils.load_video(fn_video)
print(f'Visualization for: {fn_video}')
print(cap.__sizeof__())

# LOAD PREDICTIONS
fn_predictions_pos = f'predictions_{args.model_type}_position_{args.gender}_{args.cropping}_{args.fn_video[:-4]}.csv'
df_predictions_pos = pd.read_csv(os.path.join(args.path2predictions, fn_predictions_pos))
fn_predictions_shape = f'predictions_{args.model_type}_shape_{args.gender}_{args.cropping}_{args.fn_video[:-4]}.csv'
df_predictions_shape = pd.read_csv(os.path.join(args.path2predictions, fn_predictions_shape))

# LOAD COORDINATE DATAFRAME
df_coord = pd.read_csv(os.path.join(args.path2output,
                                    f'{args.fn_video[:-4]}_coordinates.csv'))

# LOAD FEATURES
df_features = pd.read_csv(os.path.join(args.path2output,
                                       f'{args.fn_video[:-4]}_features.csv'))

# SAVE MESAURES TO CSV
df_measures = pd.read_csv(os.path.join(args.path2output,
                          f'{args.fn_video[:-4]}_measures.csv'))

# GET STIMULUS ENTIRE STRING
str_stimulus = utils.get_stimulus_string(fn_video)

# GET SYLLABLE ONSETS FROM MFA
lpc_syllables = utils.get_syllable_onset_frames_from_lpc_file(fn_video)
print(lpc_syllables)
if args.textgrid:
    onset_frames_syllables_mfa = utils.get_syllable_onset_frames_from_mfa(fn_video, lpc_syllables)
    n_syllables = len(onset_frames_syllables_mfa)
else:
    lpc_syllables = None
    onset_frames_syllables_mfa = None
    n_syllables = None


# GET SYLLABLE ONSETS from File
fn_txt = os.path.join(args.path2video, f'{os.path.basename(fn_video)}.events')
df_events = pd.read_csv(fn_txt, skiprows=1)

print(df_events)

# MARK VIDEO
#print(df_predictions_shape)
viz.mark_video(cap, fn_video,
               args.gender, args.cropping,
               str_stimulus,
               lpc_syllables,
               df_predictions_pos,
               df_predictions_shape,
               df_measures['velocity_scaled'],
               df_events['frame_number'].values,
               onset_frames_syllables_mfa,
               text_factor=1,
               show=False)

print(f'The marked video was saved to: {fn_video[:-4]}_marked_with_model_{args.gender}_{args.cropping}.avi')
