import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
from sklearn.cluster import DBSCAN
import seaborn as sns
from scipy.ndimage import uniform_filter
from scipy.ndimage import median_filter
from scipy.ndimage import maximum_filter
import numpy as np
import halo_data as hd
import matplotlib.pyplot as plt
import glob
import pandas as pd
from pathlib import Path
import matplotlib as mpl
from scipy.stats import binned_statistic_2d
from matplotlib.animation import FuncAnimation
%matplotlib qt

# %%
data = hd.getdata('F:/halo/46/depolarization')
snr = glob.glob('F:/halo/46/depolarization/snr/*_noise.csv')
classifier_folder = 'F:\\halo\\classifier'
Path(classifier_folder).mkdir(parents=True, exist_ok=True)

# %%
# date = '20190407'
date = '20180408'
file = [file for file in data if date in file][0]
df = hd.halo_data(file)

df.filter_height()
df.unmask999()
df.depo_cross_adj()

df.filter(variables=['beta_raw'],
          ref='co_signal',
          threshold=1 + df.snr_sd)

# Save before further filtering
beta_save = df.data['beta_raw'].flatten()
depo_save = df.data['depo_adj'].flatten()
co_save = df.data['co_signal'].flatten()
cross_save = df.data['cross_signal'].flatten()  # Already adjusted with bleed

df.data['classifier'] = np.zeros(df.data['beta_raw'].shape, dtype=int)

# Aerosol
aerosol = df.decision_tree(depo_thres=[None, None],
                           beta_thres=[None, -5.5],
                           v_thres=[None, None],
                           depo=df.data['depo_adj'],
                           beta=np.log10(df.data['beta_raw']),
                           v=df.data['v_raw'])

# Small size median filter to remove noise
aerosol_smoothed = median_filter(aerosol, size=11)
df.data['classifier'][aerosol_smoothed] = 10

df.filter(variables=['beta_raw', 'v_raw', 'depo_adj'],
          ref='co_signal',
          threshold=1 + 3 * df.snr_sd)
range_save = np.tile(df.data['range'],
                     df.data['beta_raw'].shape[0])

time_save = np.repeat(df.data['time'],
                      df.data['beta_raw'].shape[1])
v_save = df.data['v_raw'].flatten()  # put here to avoid noisy values at 1sd snr

# Liquid
liquid = df.decision_tree(depo_thres=[None, None],
                          beta_thres=[-5.5, None],
                          v_thres=[None, None],
                          depo=df.data['depo_adj'],
                          beta=np.log10(df.data['beta_raw']),
                          v=df.data['v_raw'])

# maximum filter to increase the size of liquid region
liquid_max = maximum_filter(liquid, size=5)
# Median filter to remove background noise
liquid_smoothed = median_filter(liquid_max, size=13)
# use snr threshold
snr = df.data['co_signal'] > (1 + 3*df.snr_sd)
lidquid_smoothed = liquid_smoothed * snr
df.data['classifier'][liquid_smoothed] = 30

# updraft - indication of aerosol zone
updraft = df.decision_tree(depo_thres=[None, None],
                           beta_thres=[None, None],
                           v_thres=[1, None],
                           depo=df.data['depo_adj'],
                           beta=np.log10(df.data['beta_raw']),
                           v=df.data['v_raw'])
updraft_smooth = median_filter(updraft, size=3)
updraft_max = maximum_filter(updraft_smooth, size=91)

# Fill the gap in aerosol zone
updraft_median = median_filter(updraft_max, size=31)

# precipitation < -1 (center of precipitation)
precipitation_1 = df.decision_tree(depo_thres=[None, None],
                                   beta_thres=[-7, None],
                                   v_thres=[None, -1],
                                   depo=df.data['depo_adj'],
                                   beta=np.log10(df.data['beta_raw']),
                                   v=df.data['v_raw'])
precipitation_1_median = median_filter(precipitation_1, size=9)

# Only select precipitation outside of aerosol zone
precipitation_1_ne = precipitation_1_median * ~updraft_median
precipitation_1_median_smooth = median_filter(precipitation_1_ne,
                                              size=3)
precipitation = precipitation_1_median_smooth

# precipitation < -0.5 (include of precipitation)
precipitation_1_low = df.decision_tree(depo_thres=[None, None],
                                       beta_thres=[-7, None],
                                       v_thres=[None, -0.5],
                                       depo=df.data['depo_adj'],
                                       beta=np.log10(df.data['beta_raw']),
                                       v=df.data['v_raw'])

# Avoid ebola infection surrounding updraft
# Useful to contain error during ebola precipitation
updraft_ebola = df.decision_tree(depo_thres=[None, None],
                                 beta_thres=[None, None],
                                 v_thres=[0.2, None],
                                 depo=df.data['depo_adj'],
                                 beta=np.log10(df.data['beta_raw']),
                                 v=df.data['v_raw'])
updraft_ebola_max = maximum_filter(updraft_ebola, size=3)

# Ebola precipitation
for _ in range(1500):
    prep_1_max = maximum_filter(precipitation, size=3)
    prep_1_max *= ~updraft_ebola_max  # Avoid updraft area
    precipitation_ = precipitation_1_low * prep_1_max
    if np.sum(precipitation) == np.sum(precipitation_):
        break
    precipitation = precipitation_

df.data['classifier'][precipitation] = 20

# Remove all aerosol above cloud or precipitation
mask_aerosol0 = df.data['classifier'] == 10
for i in np.array([20, 30]):
    mask = df.data['classifier'] == i
    mask_row = np.argwhere(mask.any(axis=1)).reshape(-1)
    mask_col = np.nanargmax(df.data['classifier'][mask_row, :] == i,
                            axis=1)
    for row, col in zip(mask_row, mask_col):
        mask[row, col:] = True
    mask_undefined = mask * mask_aerosol0
    df.data['classifier'][mask_undefined] = i

# %%
precipitation = precipitation_1_median_smooth

cmap = mpl.colors.ListedColormap(
    ['white', 'blue'])

fig, ax = plt.subplots(3, 1, figsize=(9, 9))
ax[0].pcolormesh(df.data['time'], df.data['range'],
                 np.log10(df.data['beta_raw']).T, cmap='jet',
                 vmin=-8, vmax=-4)

ax[1].pcolormesh(df.data['time'], df.data['range'],
                 df.data['v_raw'].T, cmap='jet', vmin=-2, vmax=2)
title = ax[-1].text(0.5, 0.85, "", bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5},
                    transform=ax[-1].transAxes, ha="center")

quad = ax[-1].pcolormesh(df.data['time'], df.data['range'],
                         precipitation.T, cmap=cmap)

fig.tight_layout()


def update(iter):
    global precipitation
    prep_1_max = maximum_filter(precipitation, size=3)
    prep_1_max *= ~updraft_ebola_max  # Avoid updraft area
    precipitation_ = precipitation_1_low * prep_1_max
    precipitation = precipitation_
    quad.set_array(precipitation[:-1, :-1].T.ravel())
    title.set_text(iter)
    return quad, title,


ani = FuncAnimation(fig, update, frames=200, interval=1, blit=True, repeat=False)
