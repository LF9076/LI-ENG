# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 21:19:33 2021

@author: LF
"""

import glob
import numpy as np
import mne
from mne.time_frequency import tfr_morlet
from mne.stats import permutation_cluster_1samp_test,permutation_cluster_test
import matplotlib.pyplot as plt
path = r'F:\data\HC_stimulus_epoch'
#注意，cue和stimulus的电极不一样
#cue:=['PO3','POz','PO4','Oz']
#stimulus = ['Pz','Cz']
filepath = glob.glob(path+'/*.fif')
subject_num = int(len(filepath)/2)
power_zeros = []
power_InCongruent_ave = np.array([[[]]], dtype=np.float64)
def power_(epochs):
    epochs = epochs
    freqs = np.arange(4, 30, 1) 
    n_cycles = freqs / 4.
    power = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=False, decim=5, n_jobs=1,average=True,picks = ['Pz','Cz'])
    return power
#   return :'AverageTFR'len(filepath)

for i in range(0,len(filepath),2):
    a = filepath[i]
    epochs_Congruent = mne.read_epochs(filepath[i], preload=True)
    evoked1 = epochs_Congruent.average()
    #evoked1.plot()
    epochs_Congruent.drop_channels('EOG')
    power_Congruent = power_(epochs_Congruent)
    epochs_InCongruent = mne.read_epochs(filepath[i+1], preload=True)
    epochs_InCongruent.drop_channels('EOG')
    power_InCongruent = power_(epochs_InCongruent)
    a = power_InCongruent.__sub__(power_Congruent)
    power_InCongruent_ave = np.append(power_InCongruent_ave,a.data)
    print(i)

sensor_adjacency, ch_names = mne.channels.find_ch_adjacency(epochs_InCongruent.info,'eeg')
use_idx = [ch_names.index(ch_name.replace(' ', ' '))
       for ch_name in power_InCongruent.ch_names]
sensor_adjacency = sensor_adjacency[use_idx][:, use_idx]
assert sensor_adjacency.shape == \
    (len(power_InCongruent.ch_names), len(power_InCongruent.ch_names))
adjacency = mne.stats.combine_adjacency(
    sensor_adjacency, len(power_InCongruent.freqs), len(power_InCongruent.times))

# our adjacency is square with each dim matching the data size
assert adjacency.shape[0] == adjacency.shape[1] == \
    len(power_InCongruent.ch_names) * len(power_InCongruent.freqs) * len(power_InCongruent.times)
    
power_data = power_InCongruent_ave.reshape(subject_num,len(use_idx),len(power_InCongruent.freqs),len(power_InCongruent.times))
n_permutations = 1000  # Warning: 50 is way too small for real-world analysis.
#threshold:min=-4.874401 max=7.152988
#HC:2.7 457:3
T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test (
    power_data, n_permutations=1000,
    threshold=2.7, tail=1,
    adjacency=adjacency,
    out_type='mask', verbose=True)
T_obs_plot = np.nan * np.ones_like(T_obs)
for c, p_val in zip(clusters, cluster_p_values):
    if p_val < 0.01:
        T_obs_plot[c] = T_obs[c]

vmax = np.max(np.abs(T_obs))
vmin = -vmax

plt.imshow((T_obs[0]+T_obs[1])/2, cmap=plt.cm.gray,
           extent=[-0.2,0.6,4,30],
           aspect='auto', origin='lower',vmin=0, vmax=6)

plt.imshow((T_obs_plot[0]+T_obs_plot[1])/2, cmap=plt.cm.jet,
           extent=[-0.2,0.6,4,30],
           aspect='auto', origin='lower', vmin=0, vmax=6)#+T_obs_plot[1]+T_obs_plot[2]+T_obs_plot[3])/4
plt.colorbar()
plt.xlabel('Time (ms)')
plt.ylabel('Frequency (Hz)')
plt.title(path)
