# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 12:39:23 2023

@author: WIN10
"""
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

# generate the data 
# Set the seed for reproducibility (optional)

metric_synch_m     = np.random.normal( 2, 1, 30)
metric_synch_f     = np.random.normal(1.5, 1, 30)
group_name_synch   = np.repeat('synch', 60)
gender_synch       = np.concatenate(( np.repeat('m', 30), np.repeat('f', 30)))
print(metric_synch_m,metric_synch_f)
metric_unsynch_m   = np.random.normal(1.5, 1, 30)
metric_unsynch_f   = np.random.normal(2, 1, 30)
group_name_usynch  = np.repeat('unsynch', 60)
gender_unsynch     = gender_synch  

#%% create gender varaible 

#%%
metric = np.concatenate((metric_synch_m,metric_synch_f,metric_unsynch_m,metric_unsynch_f))
gender = np.concatenate((gender_synch,gender_unsynch))
group  = np.concatenate((group_name_synch,group_name_usynch))
df     = pd.DataFrame({'metric' :metric, 'group' : group,'gender':gender}  )
#%%% visualization of the data 
sns.swarmplot(data= df , x = 'group' , y = 'metric' , hue = 'gender')
plt.show()
sns.barplot(data= df , x = 'group' , y = 'metric' , hue = 'gender')
plt.show()