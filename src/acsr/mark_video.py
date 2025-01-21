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
"""
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41582468.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41582468.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41479205.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41479205.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41582450.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41582450.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41641667.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41641667.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41905118.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41905118.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41454602.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41454602.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41589622.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41589622.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41466982.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41466982.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41453883.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41453883.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41542038.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41542038.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41585012.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41585012.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41443517.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41443517.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41479197.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41479197.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41818855.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41818855.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41599609.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41599609.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41321719.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41321719.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41535731.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41535731.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41744943.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41744943.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41485166.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41485166.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41301238.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41301238.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41391613.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41391613.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41463954.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41463954.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41612158.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41612158.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41669476.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41669476.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41893720.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41893720.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41419468.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41419468.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41576646.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41576646.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41579695.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41579695.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41685948.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41685948.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41645882.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41645882.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41414328.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41414328.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41578187.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41578187.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41589667.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41589667.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41589823.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41589823.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41466334.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41466334.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41466361.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41466361.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41364718.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41364718.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41364723.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41364723.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41252215.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41252215.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41252218.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41252218.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41446770.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41446770.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41446772.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41446772.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41484342.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41484342.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41484349.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41484349.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41488971.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41488971.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41488972.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41488972.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41588211.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41588211.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41588219.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41588219.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41364689.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41364689.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41364695.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41364695.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41486174.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41486174.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41486213.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41486213.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41595986.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41595986.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41850506.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41850506.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41822109.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41822109.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41822110.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41822110.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41492810.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41492810.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41492811.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41492811.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41534874.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41534874.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41534880.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41534880.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41468786.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41468786.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41468794.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41468794.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41468795.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41468795.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41598502.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41598502.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41598514.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41598514.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41598537.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41598537.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41821968.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41821968.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41821975.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41821975.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41821976.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41821976.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41331217.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41331217.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41331221.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41331221.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41331299.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41331299.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41424591.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41424591.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41424679.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41424679.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41424686.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41424686.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41479067.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41479067.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41479070.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41479070.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41479071.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41479071.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41659212.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41659212.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41659223.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41659223.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41659224.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41659224.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41444846.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41444846.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41444850.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41444850.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41444900.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41444900.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41542637.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41542637.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41542645.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41542645.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41542665.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41542665.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41244088.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41244088.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41244131.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41244131.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41244165.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41244165.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41244239.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41244239.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41487983.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41487983.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41488005.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41488005.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41488006.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41488006.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41488011.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41488011.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41474645.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41474645.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41474676.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41474676.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41474678.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41474678.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41474691.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41474691.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41579901.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41579901.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41580265.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41580265.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41580287.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41580287.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41584268.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41584268.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41576427.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41576427.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41576429.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41576429.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41576431.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41576431.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41576432.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41576432.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41542056.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41542056.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41542061.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41542061.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41542062.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41542062.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41542148.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41542148.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41542150.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41542150.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41591227.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41591227.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41591229.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41591229.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41591231.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41591231.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41591249.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41591249.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41591250.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41591250.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41588684.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41588684.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41588694.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41588694.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41588696.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41588696.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41588702.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41588702.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41588704.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41588704.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41878254.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41878254.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41878270.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41878270.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41878279.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41878279.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41878290.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41878290.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41878387.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41878387.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41878409.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41878409.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41463721.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41463721.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41466066.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41466066.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41466097.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41466097.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41466113.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41466113.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41506220.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41506220.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41559830.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41559830.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41694251.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41694251.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41694252.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41694252.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41694262.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41694262.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41694264.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41694264.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41694265.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41694265.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41694281.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41694281.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41694283.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41694283.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41624172.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41624172.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41624253.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41624253.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41624281.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41624281.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41624282.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41624282.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41624287.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41624287.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41624356.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41624356.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41624359.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41624359.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41624364.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41624364.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41624373.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41624373.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41517722.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41517722.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41517728.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41517728.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41517733.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41517733.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41517738.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41517738.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41517751.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41517751.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41517752.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41517752.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41517754.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41517754.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41519568.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41519568.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41519588.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41519588.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41519593.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41519593.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41586114.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41586114.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41596183.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41596183.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41638962.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41638962.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41639766.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41639766.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41720965.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41720965.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41809893.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41809893.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41809914.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41809914.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41829311.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41829311.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41856648.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41856648.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41862462.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41862462.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41550073.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41550073.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41550080.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41550080.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41550083.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41550083.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41550084.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41550084.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41550086.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41550086.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41550090.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41550090.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41550091.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41550091.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41550092.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41550092.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41550093.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41550093.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41550094.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41550094.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41357803.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41357803.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41357804.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41357804.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41357807.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41357807.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41357864.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41357864.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41357867.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41357867.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41357868.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41357868.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41357923.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41357923.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41357924.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41357924.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41358001.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41358001.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41358005.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41358005.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41573441.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41573441.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41573456.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41573456.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41573481.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41573481.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41573512.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41573512.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41573521.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41573521.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41573528.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41573528.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41573550.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41573550.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41573558.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41573558.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41573560.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41573560.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41573574.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41573574.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41573585.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41573585.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41573598.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41573598.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41580654.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41580654.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41580660.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41580660.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41580704.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41580704.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41580738.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41580738.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41580766.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41580766.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41580767.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41580767.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41580769.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41580769.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41580771.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41580771.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41580806.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41580806.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41580807.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41580807.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41580808.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41580808.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41580809.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41580809.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41580811.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41580811.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41580843.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41580843.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41592014.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41592014.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41592019.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41592019.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41592021.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41592021.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41592022.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41592022.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41592024.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41592024.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41592031.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41592031.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41592043.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41592043.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41592046.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41592046.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41592056.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41592056.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41592078.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41592078.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41592079.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41592079.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41592081.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41592081.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41592087.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41592087.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41592088.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41592088.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41592109.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41592109.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41592110.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41592110.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41577174.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41577174.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41577178.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41577178.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41577224.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41577224.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41577225.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41577225.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41577321.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41577321.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41577322.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41577322.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41577357.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41577357.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41577411.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41577411.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41577416.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41577416.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41577418.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41577418.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41577465.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41577465.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41577500.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41577500.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41577527.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41577527.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41577529.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41577529.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41578240.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41578240.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41578255.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41578255.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41578257.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41578257.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41578259.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41578259.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41248413.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41248413.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41248460.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41248460.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41248467.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41248467.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41248488.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41248488.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41248510.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41248510.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41248525.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41248525.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41248526.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41248526.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41248553.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41248553.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41249125.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41249125.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41249141.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41249141.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41848443.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41848443.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41848449.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41848449.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41848477.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41848477.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41848480.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41848480.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41848494.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41848494.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41848499.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41848499.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41848508.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41848508.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41848521.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41848521.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41848522.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41848522.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41848529.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41848529.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41333038.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41333038.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41333040.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41333040.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41333041.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41333041.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41333042.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41333042.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41333043.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41333043.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41333044.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41333044.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41410018.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41410018.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41410019.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41410019.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41410020.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41410020.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41410023.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41410023.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41410370.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41410370.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41410422.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41410422.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41410840.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41410840.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41410841.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41410841.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41410843.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41410843.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41410874.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41410874.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41410877.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41410877.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41410878.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41410878.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41469000.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41469000.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41469007.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41469007.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41469008.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41469008.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41284174.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41284174.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41284184.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41284184.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41417977.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41417977.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41417983.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41417983.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41418000.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41418000.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41418030.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41418030.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41418034.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41418034.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41418090.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41418090.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41466165.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41466165.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41466181.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41466181.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41466196.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41466196.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41466673.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41466673.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41466693.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41466693.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41466701.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41466701.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41466708.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41466708.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41466711.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41466711.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41466737.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41466737.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41466817.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41466817.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41476444.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41476444.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41476451.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41476451.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41476452.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41476452.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41476455.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41476455.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41476462.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41476462.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41476467.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41476467.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41476473.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41476473.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41476487.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41476487.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41537444.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41537444.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41537451.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41537451.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41546026.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41546026.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41546028.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41546028.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41546046.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41546046.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41673164.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41673164.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41673165.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41673165.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41673170.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41673170.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41673172.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41673172.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41674555.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41674555.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41674588.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41674588.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41674679.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41674679.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41674716.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41674716.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41674753.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41674753.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41674754.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41674754.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41674755.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41674755.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41674808.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41674808.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41674933.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41674933.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41674934.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41674934.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41674959.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41674959.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41674967.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41674967.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41674970.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41674970.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41675039.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41675039.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41675078.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41675078.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41675079.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41675079.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41675087.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41675087.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41675089.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41675089.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41675101.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41675101.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41675103.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41675103.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41675113.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41675113.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41675872.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41675872.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41675875.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41675875.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41675915.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41675915.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41675918.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41675918.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41675940.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41675940.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41675965.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41675965.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41676000.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41676000.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41676020.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41676020.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41676023.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41676023.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41403702.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41403702.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41403733.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41403733.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41403812.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41403812.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41403910.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41403910.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41403928.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41403928.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41403930.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41403930.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41403970.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41403970.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41403971.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41403971.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41403992.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41403992.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41404020.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41404020.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41404047.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41404047.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41404070.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41404070.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41404072.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41404072.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41404091.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41404091.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41404118.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41404118.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41404119.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41404119.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41404208.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41404208.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41404283.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41404283.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41404311.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41404311.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41404313.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41404313.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41404341.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41404341.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41404343.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41404343.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41404393.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41404393.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41404433.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41404433.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41404485.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41404485.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41404509.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41404509.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41404524.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41404524.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41404545.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41404545.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41404548.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41404548.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41404568.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41404568.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41404570.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41404570.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41404589.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41404589.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41404602.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41404602.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41404630.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41404630.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41404658.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41404658.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41404677.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41404677.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41404683.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41404683.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41404750.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41404750.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41404755.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41404755.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41589454.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41589454.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41589455.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41589455.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41589494.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41589494.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41589496.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41589496.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41589497.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41589497.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41589516.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41589516.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41589533.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41589533.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41589535.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41589535.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41589576.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41589576.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41589579.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41589579.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41589580.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41589580.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41589609.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41589609.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41589685.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41589685.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41589703.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41589703.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41589705.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41589705.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41589727.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41589727.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41589730.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41589730.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590088.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590088.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590089.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590089.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590097.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590097.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590098.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590098.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590100.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590100.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590106.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590106.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590108.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590108.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590117.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590117.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590124.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590124.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590130.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590130.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590131.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590131.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590133.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590133.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590142.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590142.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590144.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590144.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590145.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590145.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590146.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590146.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590149.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590149.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590178.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590178.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590179.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590179.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590189.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590189.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590190.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590190.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590192.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590192.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590194.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590194.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590195.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590195.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590199.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590199.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590200.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590200.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590206.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590206.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590208.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590208.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590209.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590209.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590212.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590212.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590213.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590213.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590214.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590214.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590217.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590217.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590229.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590229.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590233.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590233.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590235.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590235.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590237.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590237.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590244.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590244.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590247.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590247.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590248.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590248.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590256.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590256.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590258.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590258.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590259.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590259.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590267.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590267.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590268.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590268.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590277.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590277.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590278.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590278.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590291.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590291.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590300.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590300.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590308.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590308.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590310.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590310.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590313.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590313.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590315.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590315.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590320.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590320.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590323.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590323.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590324.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590324.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41590325.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41590325.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41594626.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41594626.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41594629.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41594629.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41595034.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41595034.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41595037.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41595037.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41595038.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41595038.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41595053.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41595053.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41595055.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41595055.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41595062.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41595062.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41595063.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41595063.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41595066.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41595066.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41595071.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41595071.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41595075.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41595075.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41595078.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41595078.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41595079.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41595079.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41595081.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41595081.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41595082.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41595082.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41595088.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41595088.txt
/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr/clips/common_voice_fr_41595091.mp3,/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa/common_voice_fr_41595091.txt

"""