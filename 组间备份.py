# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 01:04:45 2021

@author: LF
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 17:13:34 2021

@author: LF
"""

import glob
import numpy as np
import mne
from mne.time_frequency import tfr_morlet
from mne.stats import permutation_cluster_1samp_test
import matplotlib.pyplot as plt
#path = r'F:\data\457_cue_epoch'
#path = r'F:\data\HC_cue_epoch'
#path = r'F:\data\457_stimulus_epoch'
path = r'F:\data\HC_stimulus_epoch'
filepath = glob.glob(path+'/*.fif')
power_zeros = []
def power_(epochs):
    epochs = epochs
    freqs = np.arange(4, 30, 1) 
    n_cycles = freqs / 2.
    power = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=False, decim=5, n_jobs=1,average=False,picks = ['Pz','Cz','Fz'])
    return power
#   return :'AverageTFR'
for i in range(0,2,2):
    a = filepath[i]
    epochs_Congruent = mne.read_epochs(filepath[i], preload=True)
    epochs_Congruent.drop_channels('EOG')
    power_Congruent = power_(epochs_Congruent)
    epochs_InCongruent = mne.read_epochs(filepath[i+1], preload=True)
    epochs_InCongruent.drop_channels('EOG')
    power_InCongruent = power_(epochs_InCongruent)
    epochs_power = power_InCongruent.data
sensor_adjacency, ch_names = mne.channels.find_ch_adjacency(epochs_InCongruent.info,'eeg')
use_idx = [ch_names.index(ch_name.replace(' ', ''))
       for ch_name in power_InCongruent.ch_names]
sensor_adjacency = sensor_adjacency[use_idx][:, use_idx]
assert sensor_adjacency.shape == \
    (len(power_InCongruent.ch_names), len(power_InCongruent.ch_names))
assert epochs_power.data.shape == (len(epochs_power),
                                   len(power_InCongruent.ch_names),
                                   len(power_InCongruent.freqs),
                                   len(power_InCongruent.times))
adjacency = mne.stats.combine_adjacency(
    sensor_adjacency, len(power_InCongruent.freqs), len(power_InCongruent.times))

# our adjacency is square with each dim matching the data size
assert adjacency.shape[0] == adjacency.shape[1] == \
    len(power_InCongruent.ch_names) * len(power_InCongruent.freqs) * len(power_InCongruent.times)
n_permutations = 5000  # Warning: 50 is way too small for real-world analysis.
T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
    epochs_power, n_permutations=n_permutations,
    threshold=14 , tail=0,
    adjacency=adjacency,
    out_type='mask', verbose=True)
T_obs_plot = np.nan * np.ones_like(T_obs)
for c, p_val in zip(clusters, cluster_p_values):
    if p_val <= 0.05:
        T_obs_plot[c] = T_obs[c]

# Just plot one channel's data
ch_idx, f_idx, t_idx = np.unravel_index(
    np.nanargmax(np.abs(T_obs_plot)), epochs_power.shape[1:])
# ch_idx = tfr_epochs.ch_names.index('MEG 1332')  # to show a specific one

vmax = np.max(np.abs(T_obs))
vmin = -vmax
plt.subplot(2, 1, 1)
plt.imshow(T_obs[ch_idx], cmap=plt.cm.gray,
           aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
plt.imshow(T_obs_plot[ch_idx], cmap=plt.cm.RdBu_r,
           aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
plt.colorbar()
plt.xlabel('Time (ms)')
plt.ylabel('Frequency (Hz)')
plt.title(f'Induced power')