#!/usr/bin/env python
# coding: utf-8

import os
import textgrids
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
import pylab
import parselmouth

path2stimuli = '../stimuli/sentences/'
path2figures = '../stimuli/sentences/png'
f_textgrid = 'sent_01.TextGrid'
f_text = 'sent_01.txt'
f_wav = 'sent_01.wav'
fn_fig = 'sent_01.png'

grid = textgrids.TextGrid(os.path.join(path2stimuli, 'mfa_output', f_textgrid))
phones = grid['phones']
words = grid['words']

sample_rate, samples = wavfile.read(os.path.join(path2stimuli, 'wav', f_wav))
samples = samples[:, 0]

fig, axs = plt.subplots(2, 1, figsize=(20,10))
times_sec = np.asarray(range(len(samples)))/sample_rate

axs[0].plot(times_sec, samples/max(abs(samples)))


snd = parselmouth.Sound(os.path.join(path2stimuli, 'wav', f_wav))
dynamic_range = 70

intensity = snd.to_intensity()
spectrogram = snd.to_spectrogram()
X, Y = spectrogram.x_grid(), spectrogram.y_grid()
sg_db = 10 * np.log10(spectrogram.values)
axs[1].pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap='afmhot')
axs[1].set_ylim([spectrogram.ymin, spectrogram.ymax])
axs[1].set_xlabel("Time [s]", fontsize=14)
axs[1].set_ylabel("Frequency [Hz]", fontsize=14)

#axs[1].twinx()
#axs[1].plot(intensity.xs(), intensity.values.T, linewidth=3, color='w')
#axs[1].plot(intensity.xs(), intensity.values.T, linewidth=1)
#axs[1].grid(False)
#axs[1].set_ylim(0)
#axs[1].set_ylabel("intensity [dB]")
#axs[1].set_xlim([snd.xmin, snd.xmax])

#ipylab.specgram(samples, NFFT=80, Fs=16000, noverlap=40)

phones_str = []
phones_times = []
for phone in phones:
    axs[1].axvline(phone.xmin, ymax=8000, color='k', ls='--')
    # plt.text(phone.xmin, 7500, phone.text, verticalalignment='center')
    phones_str.append(phone.text)
    phones_times.append(phone.xmin)

for word in words:
    if not word.text in ['sil', 'sp']:
        axs[0].axvline(word.xmin, ymax=8000, color='r', ls='--')
        axs[0].text(word.xmin, 1.1, word.text, verticalalignment='center', fontsize=16)
        #axs[1].axvline(word.xmin, ymax=8000, color='r', ls='--')
        #axs[1].text(word.xmin, 5250, word.text, verticalalignment='center', fontsize=16)

plt.setp(axs[0], ylabel='Signal', xlim=[0, max(times_sec)], ylim=[-1, 1])
axs[0].set_xlabel('Time [sec]', fontsize=14)
axs[0].set_ylabel('Acoustic Waveform', fontsize=14)
plt.setp(axs[1], xlim=[0, max(times_sec)], xticks=phones_times, xticklabels=phones_str)
plt.setp(axs[1].get_xticklabels(), fontsize=14)
#axs[1].set_ylabel('Frequency [Hz]', fontsize=14)

fn_fig = os.path.join(path2figures, fn_fig)
plt.savefig(fn_fig)
print(f'Figure as saved to: {fn_fig}')

