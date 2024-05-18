# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 13:35:24 2023

@author: WIN10
"""
# import packages 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import savemat
from scipy import signal
from ordpy import renyi_complexity_entropy as RE
from ordpy import renyi_entropy
from ordpy import permutation_entropy
from neurodsp.burst import detect_bursts_dual_threshold as DBDT
from neurodsp.burst import compute_burst_stats
import scipy.stats
from scipy.stats import mannwhitneyu
from scipy.stats import wilcoxon

sns.set_theme()

#%%% define functions 


def band_filter(data, low, high, rate):
    b, a = signal.butter(4, [low, high], btype='band', fs=rate)
    return signal.filtfilt(b, a, data)


def notch_filter(data, freq, rate):
    b, a = signal.iirnotch(freq, 30, fs=rate)
    return signal.filtfilt(b, a, data)


def filter_data(data):
    temp = band_filter(data, 1, 80, 250)
    temp = notch_filter(temp, 50, 250)
    return notch_filter(temp, 60, 250)


def read_file(path):
    frame = pd.read_csv(path, delimiter=',\s+', comment='%')
    data = frame.loc[:, ['EXG Channel 2', 'EXG Channel 4']].values.T
    for i in range(len(data)): data[i] = filter_data(data[i])
    return data


def split_signal(data, meta, offset=0):
    # what is offeset in this function some recordings have not started from the 0 time 
    start = offset * 250
    output, target = [], []
    for level, duration in meta:
        end = start + duration * 250
        output.append(data[:, start:end])
        target.append(level)
        start = end
    return output, np.array(target)

def plot_data(data):
    fig, ax = plt.subplots(1, 1, figsize=(18, 5))
    ax.plot(data)
    fig.tight_layout()

def Conditioner(data , levels , duration= 120):
    Low, Medium , High = [], [] ,[]

    for k in range(len(levels)):
        if levels[k]<1 and np.shape(data[k])[1]>= 250*duration:
            Low.append(data[k][:,:duration*250])
        elif 1 <= levels[k] < 2 and np.shape(data[k])[1]>= 250*duration:
            Medium.append(data[k][:,:duration*250])
        elif levels[k]>=2 and np.shape(data[k])[1]>= 250*duration:
            High.append(data[k][:,:duration*250])
    return Low , Medium , High
# 
def WindowingBDT(data, window = 5,ThrH=2, ThrL= 1, duration= 0.5 ):
    d = []
    for j in data:
        for i in range(0,24):
            d.append(DBDT(j[i*window*250:(i+1)*window*250], dual_thresh=(ThrH,ThrL), fs=250, min_burst_duration=duration, avg_type='median'))

    return np.array(d).reshape(10,24,1250)

def StatDBT(data):

    a, f = [] , []
    for i in data:
        for j in i:
            f.append(compute_burst_stats(j, 250))
    f = np.array(f).reshape(10,24)

    for m in f:
        for n in m:

            a.append(n['n_bursts'])
    return  np.sum(np.array(a).reshape(10,24), axis =1)
#%%% load the data 

path = 'D:/science/zebra_finch_eeg/zebra-finch_eeg/IsofluraneData/FinchOne/OpenBCI-RAW-2023-04-16_16-53-40.txt'
meta = [(1, 180), (.6, 120), (.4, 120), (.6, 120), (1, 150), (1.5, 150), (2, 150), (.2, 150)]
data1 = read_file(path)
signal1, level1 = split_signal(data1, meta)

path = 'D:/science/zebra_finch_eeg/zebra-finch_eeg/IsofluraneData/FichTwo/OpenBCI-RAW-2023-04-17_17-05-56.txt'
meta = [(1.5, 150), (1, 80), (1.5, 190), (2, 120), (2.5, 160), (2, 120), (1.5, 128), (1, 132), (.6, 126), (.4, 124)]
data2 = read_file(path)
signal2, level2 = split_signal(data2, meta, 480)

path = 'D:/science/zebra_finch_eeg/zebra-finch_eeg/IsofluraneData/FinchThree-1,2/OpenBCI-RAW-2023-05-23_11-33-40.txt'
meta = [(2, 150), (1, 80), (1.5, 190), (2, 120), (2.5, 160), (2, 120), (1.5, 128), (1, 132), (.6, 126), (.4, 124)]
data31 = read_file(path)
signal31, level31 = split_signal(data31, meta)

path = 'D:/science/zebra_finch_eeg/zebra-finch_eeg/IsofluraneData/FinchThree-1,2/OpenBCI-RAW-2023-05-23_11-57-18.txt'
meta = [(2, 120)]
data32 = read_file(path)
signal32, level32 = split_signal(data32, meta)

path = 'D:/science/zebra_finch_eeg/zebra-finch_eeg/IsofluraneData/FinchFour-1/OpenBCI-RAW-2023-05-23_15-47-21.txt'
meta = [(1.5, 120), (1, 90), (1.5, 120), (2, 120), (2.5, 120)]
data41 = read_file(path)
signal41, level41 = split_signal(data41, meta, offset = 150)

path = 'D:/science/zebra_finch_eeg/zebra-finch_eeg/IsofluraneData/FinchFour_2/OpenBCI-RAW-2023-05-23_16-03-43.txt'
meta = [(2, 120), (1.5, 120), (1, 120), (.6, 120), (.4, 120)]
data42 = read_file(path)
signal42, level42 = split_signal(data42, meta)

#%%


s1  = Conditioner(signal1 , level1 , duration= 120)
s2  = Conditioner(signal2 , level2 , duration= 120)
s31 = Conditioner(signal31, level31, duration= 120)
s32 = Conditioner(signal32, level32, duration= 120)
s41 = Conditioner(signal41, level41, duration= 120)
s42 = Conditioner(signal42, level42, duration= 120)

#%%

LowC1 = np.concatenate((s1[0],s2[0],s31[0],s42[0]),axis = 0)[:10,0]
MediumC1 = np.concatenate((s1[1],s2[1],s31[1], s41[1],s42[1]),axis = 0)[:10,0]
HighC1 = np.concatenate((s1[2],s2[2],s31[2],s32[2],s41[2],s42[2]),axis = 0)[:10,0]

LowC2 = np.concatenate((s1[0],s2[0],s31[0],s42[0]),axis = 0)[:10,1]
MediumC2 = np.concatenate((s1[1],s2[1],s31[1], s41[1],s42[1]),axis = 0)[:10,1]
HighC2 = np.concatenate((s1[2],s2[2],s31[2],s32[2],s41[2],s42[2]),axis = 0)[:10,1]

#%%


NLC1 = StatDBT(WindowingBDT(LowC1,ThrH=2, ThrL= 1, duration = 1))
NMC1 = StatDBT(WindowingBDT(MediumC1,ThrH=2, ThrL= 1,duration = 1))
NHC1 = StatDBT(WindowingBDT(HighC1,ThrH=2, ThrL= 1,duration = 1))
print(sum(NLC1), sum(NMC1), sum(NHC1) )
print(NLC1, NMC1, NHC1 )

#%%

print(mannwhitneyu(NLC1,NMC1))
print(mannwhitneyu(NLC1,NHC1))
print(mannwhitneyu(NHC1,NMC1))

#%%%
NLC2 = StatDBT(WindowingBDT(LowC2,ThrH=2, ThrL= 1,duration = 1))
NMC2 = StatDBT(WindowingBDT(MediumC2,ThrH=2, ThrL= 1,duration = 1))
NHC2 = StatDBT(WindowingBDT(HighC2,ThrH=2, ThrL= 1,duration = 1))

print(sum(NLC2), sum(NMC2), sum(NHC2) )
#%%
print(mannwhitneyu(NLC2,NMC2))
print(mannwhitneyu(NLC2,NHC2))
print(mannwhitneyu(NHC2,NMC2))
#%%
signal31[6][0][0:1250]

#%%%
def WindoBDT(data, window = 5, ThrH=20, ThrL= 1, duration= 0.5 ):
    d = []
    for i in range (0, int(len(data)/ 1250)):
        d.append(DBDT(data[i*window*250:(i+1)*window*250], dual_thresh=(ThrH,ThrL),
                      fs=250, min_burst_duration=duration, avg_type='median'))
    
    return np.array(d)
#%%
R31 = np.asarray(WindoBDT(signal31[6][0],ThrH=5, ThrL= 5))
R32 = np.asarray(WindoBDT(signal31[9][0],ThrH=5, ThrL= 5))

R31 = R31.flatten('C')
R32 = R32.flatten('C')
#%%
fig, ax = plt.subplots(1, 1, figsize=(18, 5))
plt.title('100 Second of 2.5 isofluran', size = 20)
plt.xlabel('Time', size = 20)
plt.ylabel('Amplitude', size = 20)

plt.plot(signal31[4][0][:25000])
plt.plot( R31[:25000]*100)









