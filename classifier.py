from scipy.ndimage import uniform_filter
from scipy.ndimage import median_filter
from scipy.ndimage import maximum_filter
import numpy as np
import halo_data as hd
import matplotlib.pyplot as plt
import glob
import pandas as pd
%matplotlib qt

# %%
data = hd.getdata('F:/halo/46/depolarization')
snr = glob.glob('F:/halo/46/depolarization/snr/*_noise.csv')

# %%
date = '20180611'
file = [file for file in data if date in file][0]
df = hd.halo_data(file)

# %%
noise_csv = [noise_file for noise_file in snr
             if df.filename in noise_file][0]
noise = pd.read_csv(noise_csv, usecols=['noise'])
thres = 1 + 3 * np.std(noise['noise'])

# %%
df.filter_height()
df.unmask999()

# %%
df.filter(variables=['beta_raw', 'v_raw', 'depo_raw'],
          ref='co_signal',
          threshold=1 + 3 * (np.std(noise['noise'])/1))

# %%
df.data['classifier'] = np.zeros(df.data['beta_raw'].shape, dtype=int)

# Aerosol
aerosol = df.decision_tree(depo_thres=[None, None],
                           beta_thres=[None, -5.5],
                           v_thres=[None, None],
                           depo=df.data['depo_raw'],
                           beta=np.log10(df.data['beta_raw']),
                           v=df.data['v_raw'])

# Small size median filter to remove noise
aerosol_smoothed = median_filter(aerosol, size=3)
df.data['classifier'][aerosol_smoothed] = 3

# Liquid
liquid = df.decision_tree(depo_thres=[None, None],
                          beta_thres=[-5.5, None],
                          v_thres=[None, None],
                          depo=df.data['depo_raw'],
                          beta=np.log10(df.data['beta_raw']),
                          v=df.data['v_raw'])

# maximum filter to increase the size of liquid region
liquid_smoothed = maximum_filter(liquid, size=5)
# Median filter to remove background noise
liquid_smoothed = median_filter(liquid_smoothed, size=13)

df.data['classifier'][liquid_smoothed] = 4

# Precipitation < -1.5m/s
precipitation_15 = df.decision_tree(depo_thres=[None, None],
                                    beta_thres=[None, None],
                                    v_thres=[None, -1.5],
                                    depo=df.data['depo_raw'],
                                    beta=np.log10(df.data['beta_raw']),
                                    v=df.data['v_raw'])

precipitation_15 = median_filter(precipitation_15, size=(45, 21))
precipitation_15 = maximum_filter(precipitation_15, size=(99, 99))
# Precipitation < -1m/s
precipitation_1 = df.decision_tree(depo_thres=[None, None],
                                   beta_thres=[None, None],
                                   v_thres=[None, -1],
                                   depo=df.data['depo_raw'],
                                   beta=np.log10(df.data['beta_raw']),
                                   v=df.data['v_raw'])

# precipitation_1 = median_filter(precipitation_1, size=9)
precipitation = precipitation_1 * precipitation_15


# precipitation_smoothed = median_filter(precipitation_smoothed, size=9)
# precipitation_smoothed = maximum_filter(precipitation_smoothed, size=(45, 45))
df.data['classifier'][precipitation] = 1

# %%
fig, ax = plt.subplots(3, 1, sharex=True)
ax[0].pcolormesh(df.data['time'], df.data['range'],
                 precipitation_15.T)
ax[1].pcolormesh(df.data['time'], df.data['range'],
                 precipitation_1.T)
ax[2].pcolormesh(df.data['time'], df.data['range'],
                 precipitation.T)
# %%
df.cbar_lim['classifier'] = [0, 4]
df.plot(variables=['beta_raw', 'v_raw', 'classifier'])
