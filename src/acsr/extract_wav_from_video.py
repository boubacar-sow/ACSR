#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 10:28:31 2022

@author: yl254115
"""

import os
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--fn-video', default='sent_01.mp4')
parser.add_argument('--path2video', default=os.path.join('..', 'stimuli', 'sentences', 'mp4'))
args = parser.parse_args()

# INPUT
fn_video = os.path.join(args.path2video, args.fn_video)

# OUTPUT
fn_wav = args.fn_video+'.wav'
dir_wav = os.path.join(args.path2video, 'audio_only')
os.makedirs(dir_wav, exist_ok=True)
fn_wav = os.path.join(dir_wav, fn_wav)

# LAUNCH
command = f"ffmpeg -i {fn_video} -ab 160k -ac 2 -ar 44100 -vn {fn_wav}"
subprocess.call(command, shell=True)
