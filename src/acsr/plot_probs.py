#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 21:18:28 2022

@author: yl254115
"""

import argparse
import os
import pandas as pd

from utils import load_model, load_video, compute_velocity
from viz import plot_predictions

parser = argparse.ArgumentParser()
parser.add_argument('--model-type', choices=['rf', 'lr', 'rc', 'gb'],
                    help = 'rf:random-forest; lr:logisitic-regrssion',
                    default='rf')
parser.add_argument('--fn-video', default='sent_01.mp4')
parser.add_argument('--path2video', default=os.path.join('..', 'data',
                                                         'test_videos'))
parser.add_argument('--path2predictions', default=os.path.join('..',
                                                               'output')) 
parser.add_argument('--show-video', action='store_true', default=False)
args = parser.parse_args()

# LOAD PREDICTIONS
fn_predictions_pos = f'predictions_{args.model_type}_position_{args.fn_video[:-4]}.csv'
df_predictions_pos = pd.read_csv(os.path.join(args.path2predictions, fn_predictions_pos))
fn_predictions_shape = f'predictions_{args.model_type}_shape_{args.fn_video[:-4]}.csv'
df_predictions_shape = pd.read_csv(os.path.join(args.path2predictions, fn_predictions_shape))

df_coord = pd.read_csv(os.path.join(args.path2predictions,
                                    f'{args.fn_video[:-4]}_coordinates.csv'))

velocity, acceleration = compute_velocity(df_coord, 'r_hand9',
                                          fn=f'../output/velocity_{args.fn_video}')



print(df_predictions_pos, df_predictions_shape)
fig, ax = plot_predictions(df_predictions_pos, df_predictions_shape, velocity)

os.makedirs(os.path.join('..', 'figures'), exist_ok=True)

fn_fig = os.path.join('..', 'figures',
                      'probs_' + args.fn_video[:-4] + '.png')
fig.savefig(fn_fig)
print(f'Figure saved to: {fn_fig}')
