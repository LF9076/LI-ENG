# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 00:10:43 2021

@author: LF
"""
import glob
import numpy as np
import mne
from mne.time_frequency import tfr_morlet
path = r'F:\data\457_cue_epoch'
filepath = glob.glob(path+'/*.fif')
power_zeros = []
def power_(epochs):
    epochs = epochs
    freqs = np.arange(4, 30, 1) 
    n_cycles = freqs / 2
    power, itc = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, decim=1, n_jobs=1,picks=['POz','Oz','O1','O2'])
    return power
for i in range(0,2,2):
    epochs_double = mne.read_epochs(filepath[i], preload=False)
    a11 = epochs_double.info
    print(epochs_double.info)
    power_double = power_(epochs_double)
    epochs_no = mne.read_epochs(filepath[i+1], preload=False)
    power_no = power_(epochs_no)
    AlertData = power_double.data - power_no.data
    #power_double.data = AlertData
    power_zeros.append(AlertData)
a = 0
#del power_zeros[4]
#del power_zeros[27]
for index in power_zeros:
        a+=index
'''
power_double.plot(picks=['PO3','POz','PO4','Oz'],tmin=0, tmax=0.6,
                  baseline=(-0.1, 0),mode='mean', title='double',combine='mean')
power_no.plot(picks=['PO3','POz','PO4','Oz'],tmin=0, tmax=0.6,
                  baseline=(-0.1, 0),mode='mean', title='no',combine='mean')
AlertData = power_double.data - power_no.data
'''
power_double.data = AlertData/(len(power_zeros))
power_double.plot(picks=['POz','Oz','O1','O2'],tmin=-0.2, tmax=0.6,
                  baseline=(-0.1, 0),mode='mean', title='athlete',combine='mean',
                  vmin = -2e-11,vmax = 2e-11)#vmin = -5e-11,vmax = 5e-11
