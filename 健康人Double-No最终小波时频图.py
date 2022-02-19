
import glob
import numpy as np
import mne
from mne.time_frequency import tfr_morlet
path = r'F:\data\HC_stimulus_epoch'
filepath = glob.glob(path+'/*.fif')
power_zeros = []
def power_(epochs):
    epochs = epochs
    freqs = np.arange(4, 30, 1) 
    n_cycles = freqs / 4
    power, itc = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, decim=2, n_jobs=1,picks=['POz','Oz','O1','O2'])
    return power
for i in range(0,len(filepath),2):
    epochs_double = mne.read_epochs(filepath[i], preload=False)
    power_double = power_(epochs_double)
    epochs_no = mne.read_epochs(filepath[i+1], preload=False)
    power_no = power_(epochs_no)
    a = power_double.__sub__(power_no)
    power_zeros.append(a)
    b = a.plot(picks=['POz','Oz','O1','O2'],tmin=-0.2,
               tmax=0.6,baseline=(-0.1, 0),mode='mean', title=str(i),
               combine='mean',vmin = -2e-10,vmax = 2e-10)
c = 0
for index in power_zeros:
        c+=index.data
power_double.data = c/(len(power_zeros))
power_double.plot(picks=['POz','Oz','O1','O2'],tmin=-0, tmax=0.8,
                  baseline=(-0.2, 0),mode='mean', title=path,combine='mean',
                  vmin = -2e-10,vmax = 2e-10)#vmin = -5e-11,vmax = 5e-11
'''
power_double.plot(picks=['PO3','POz','PO4','Oz'],tmin=0, tmax=0.6,
                  baseline=(-0.1, 0),mode='mean', title='double',combine='mean')
power_no.plot(picks=['PO3','POz','PO4','Oz'],tmin=0, tmax=0.6,
                  baseline=(-0.1, 0),mode='mean', title='no',combine='mean')
AlertData = power_double.data - power_no.data
'''
'O1','O2'
'''
power_double.data = a/(len(power_zeros))
power_double.plot(picks=['Oz'],tmin=-0.1, tmax=0.5,
                  baseline=(-0.1, 0),mode='mean', title='2',combine='mean',
                  vmin = -2e-10,vmax = 2e-10)
power_double.plot(picks=['O1'],tmin=-0.1, tmax=0.5,
                  baseline=(-0.1, 0),mode='mean', title='3',combine='mean',
                  vmin = -2e-10,vmax = 2e-10)
power_double.plot(picks=['O2'],tmin=-0.1, tmax=0.5,
                  baseline=(-0.1, 0),mode='mean', title='4',combine='mean',
                  vmin = -2e-10,vmax = 2e-10)
'''