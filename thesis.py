import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import datetime
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import copy
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from scipy.ndimage import maximum_filter
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
import os
from scipy.stats import binned_statistic_2d
from matplotlib.colors import LogNorm
import calendar
%matplotlib qt


# %%
path = r'C:\Users\vietl\Desktop\Thesis\Img'

# %%
mat = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

fig, ax = plt.subplots(1, 2, figsize=(6, 3), sharey=True)
ax[0].pcolormesh(np.flip(mat, axis=0), cmap='Reds')

for i, row in enumerate(np.flip(mat.T, axis=1)):
    for ii, col in enumerate(row):
        ax[0].text(i+0.5, ii+0.5, col, horizontalalignment='center',
                   verticalalignment='center')
ax[0].set_xlim([0, 10])
ax[0].set_ylim([0, 10])
ax[0].set_xticks(np.arange(10))
ax[0].set_yticks(np.arange(10))
ax[0].grid()
ax[0].xaxis.set_visible(False)
ax[0].yaxis.set_visible(False)
ax[1].pcolormesh(np.flip(median_filter(mat, size=3), axis=0), cmap='Reds')

for i, row in enumerate(np.flip(median_filter(mat, size=3).T, axis=1)):
    for ii, col in enumerate(row):
        ax[1].text(i+0.5, ii+0.5, col, horizontalalignment='center',
                   verticalalignment='center')
ax[1].set_xlim([0, 10])
ax[1].set_ylim([0, 10])
ax[1].set_xticks(np.arange(10))
ax[1].set_yticks(np.arange(10))
ax[1].grid()
ax[1].xaxis.set_visible(False)
ax[1].yaxis.set_visible(False)

fig.tight_layout()
fig.savefig(path + '/median_filter.png', bbox_inches='tight')

###############################################################
# %%
###############################################################

data = hd.getdata('F:/halo/46/depolarization')

# %%
date = '20180812'
file = [file for file in data if date in file][0]
df = hd.halo_data(file)

df.filter_height()
df.unmask999()
df.depo_cross_adj()

# %%
m_ = ((df.data['time'] > 5.70) &
      (df.data['time'] < 5.85)) | ((df.data['time'] > 6.20) &
                                   (df.data['time'] < 6.6))
m_ = m_ | ((df.data['time'] > 17.7) &
           (df.data['time'] < 17.9)) | ((df.data['time'] > 7.02) &
                                        (df.data['time'] < 7.1))

temp_co = df.data['co_signal'].copy()
lol = np.isnan(np.log10(df.data['beta_raw']))
lol[~m_, :] = False
temp_co[lol] = np.nan


# %%
fig, axes = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(6, 5))
p = axes[0].pcolormesh(df.data['time'], df.data['range'],
                       np.log10(df.data['beta_raw']).T, cmap='jet',
                       vmin=-8, vmax=-4)
axes[1].yaxis.set_major_formatter(hd.m_km_ticks())
axes[0].set_xlim([0, 24])
cbar = fig.colorbar(p, ax=axes[0], fraction=0.05)
cbar.ax.set_ylabel('Beta [' + df.units.get('beta_raw', None) + ']', rotation=90)
axes[0].set_ylabel('Range [km, a.g.l]')
p = axes[1].pcolormesh(df.data['time'], df.data['range'],
                       df.data['v_raw'].T, cmap='jet',
                       vmin=-2, vmax=2)
cbar = fig.colorbar(p, ax=axes[1], fraction=0.05)
cbar.ax.set_ylabel('Velocity [' + df.units.get('v_raw', None) + ']', rotation=90)
axes[1].set_ylabel('Range [km, a.g.l]')

p = axes[2].pcolormesh(df.data['time'], df.data['range'],
                       temp_co.T, cmap='jet',
                       vmin=0.995, vmax=1.005)
cbar = fig.colorbar(p, ax=axes[2], fraction=0.05)
cbar.ax.set_ylabel('co-SNR + 1', rotation=90)
axes[2].set_ylabel('Range [km, a.g.l]')
axes[2].set_xlabel('Time [UTC - hour]')
fig.tight_layout()
fig.savefig(path + '/raw_data.png', bbox_inches='tight')

# %%
df.filter(variables=['beta_raw'],
          ref='co_signal',
          threshold=1 + df.snr_sd)

# Save before further filtering
beta_save = df.data['beta_raw'].flatten()
depo_save = df.data['depo_adj'].flatten()
co_save = df.data['co_signal'].flatten()
cross_save = df.data['cross_signal'].flatten()  # Already adjusted with bleed

df.data['classifier'] = np.zeros(df.data['beta_raw'].shape, dtype=int)

log_beta = np.log10(df.data['beta_raw'])
# Aerosol
aerosol = log_beta < -5.5

# Small size median filter to remove noise
aerosol_smoothed = median_filter(aerosol, size=11)
# Remove thin bridges, better for the clustering
aerosol_smoothed = median_filter(aerosol_smoothed, size=(15, 1))

# %%
df.data['classifier'][aerosol_smoothed] = 10

# %%
cmap = mpl.colors.ListedColormap(
    ['white', '#2ca02c', 'blue', 'red', 'gray'])
boundaries = [0, 10, 20, 30, 40, 50]
norm = mpl.colors.BoundaryNorm(boundaries, cmap.N, clip=True)
fig, ax = plt.subplots(figsize=(6, 2))
p = ax.pcolormesh(df.data['time'], df.data['range'],
                  df.data['classifier'].T,
                  cmap=cmap, norm=norm)
ax.yaxis.set_major_formatter(hd.m_km_ticks())
ax.set_ylabel('Range [km, a.g.l]')
ax.set_xlabel('Time [UTC - hour]')
cbar = fig.colorbar(p, ax=ax, ticks=[5, 15, 25, 35, 45])
cbar.ax.set_yticklabels(['Background', 'Aerosol',
                         'Precipitation', 'Clouds', 'Undefined'])
fig.tight_layout()
fig.savefig(path + '/algorithm_aerosol.png', bbox_inches='tight')

# %%
df.filter(variables=['beta_raw', 'v_raw', 'depo_adj'],
          ref='co_signal',
          threshold=1 + 3 * df.snr_sd)
log_beta = np.log10(df.data['beta_raw'])

range_save = np.tile(df.data['range'],
                     df.data['beta_raw'].shape[0])

time_save = np.repeat(df.data['time'],
                      df.data['beta_raw'].shape[1])
v_save = df.data['v_raw'].flatten()  # put here to avoid noisy values at 1sd snr

# Liquid
liquid = log_beta > -5.5

# maximum filter to increase the size of liquid region
liquid_max = maximum_filter(liquid, size=5)
# Median filter to remove background noise
liquid_smoothed = median_filter(liquid_max, size=13)

df.data['classifier'][liquid_smoothed] = 30

# %%
fig, ax = plt.subplots(figsize=(6, 2))
p = ax.pcolormesh(df.data['time'], df.data['range'],
                  df.data['classifier'].T,
                  cmap=cmap, norm=norm)
ax.yaxis.set_major_formatter(hd.m_km_ticks())
ax.set_ylabel('Range [km, a.g.l]')
ax.set_xlabel('Time [UTC - hour]')
cbar = fig.colorbar(p, ax=ax, ticks=[5, 15, 25, 35, 45])
cbar.ax.set_yticklabels(['Background', 'Aerosol',
                         'Precipitation', 'Clouds', 'Undefined'])
fig.tight_layout()
fig.savefig(path + '/algorithm_cloud.png', bbox_inches='tight')

# %%
# updraft - indication of aerosol zone
updraft = df.data['v_raw'] > 1
updraft_smooth = median_filter(updraft, size=3)
updraft_max = maximum_filter(updraft_smooth, size=91)

# Fill the gap in aerosol zone
updraft_median = median_filter(updraft_max, size=31)

# precipitation < -1 (center of precipitation)
precipitation_1 = (log_beta > -7) & (df.data['v_raw'] < -1)

precipitation_1_median = median_filter(precipitation_1, size=9)

# Only select precipitation outside of aerosol zone
precipitation_1_ne = precipitation_1_median * ~updraft_median
precipitation_1_median_smooth = median_filter(precipitation_1_ne,
                                              size=3)
precipitation = precipitation_1_median_smooth

# precipitation < -0.5 (include all precipitation)
precipitation_1_low = (log_beta > -7) & (df.data['v_raw'] < -0.5)

# Avoid ebola infection surrounding updraft
# Useful to contain error during ebola precipitation
updraft_ebola = df.data['v_raw'] > 0.2
updraft_ebola_max = maximum_filter(updraft_ebola, size=3)

# %%
fig, ax = plt.subplots(2, 1, figsize=(6, 4), sharex=True, sharey=True)
temp = df.data['classifier'].copy()

# Ebola precipitation
for i in range(1500):
    if i == 1:
        temp[precipitation] = 20
        p = ax[0].pcolormesh(df.data['time'], df.data['range'],
                             temp.T, cmap=cmap, norm=norm)
        ax[0].yaxis.set_major_formatter(hd.m_km_ticks())
        ax[0].set_ylabel('Range [km, a.g.l]')
        cbar = fig.colorbar(p, ax=ax[0], ticks=[5, 15, 25, 35, 45])
        cbar.ax.set_yticklabels(['Background', 'Aerosol',
                                 'Precipitation', 'Clouds', 'Undefined'])
    prep_1_max = maximum_filter(precipitation, size=3)
    prep_1_max *= ~updraft_ebola_max  # Avoid updraft area
    precipitation_ = precipitation_1_low * prep_1_max
    if np.sum(precipitation) == np.sum(precipitation_):
        break
    precipitation = precipitation_

temp[precipitation] = 20
p = ax[1].pcolormesh(df.data['time'], df.data['range'],
                     temp.T, cmap=cmap, norm=norm)
ax[1].yaxis.set_major_formatter(hd.m_km_ticks())
cbar = fig.colorbar(p, ax=ax[1], ticks=[5, 15, 25, 35, 45])
cbar.ax.set_yticklabels(['Background', 'Aerosol',
                         'Precipitation', 'Clouds', 'Undefined'])
ax[1].set_ylabel('Range [km, a.g.l]')
ax[1].set_xlabel('Time [UTC - hour]')
fig.tight_layout()
fig.savefig(path + '/algorithm_precipitation.png', bbox_inches='tight')

# %%
df.data['classifier'][precipitation] = 20

# Remove all aerosol above cloud or precipitation
mask_aerosol0 = df.data['classifier'] == 10
for i in np.array([20, 30]):
    if i == 20:
        mask = df.data['classifier'] == i
    else:
        mask = log_beta > -5
        mask = maximum_filter(mask, size=5)
        mask = median_filter(mask, size=13)
    mask_row = np.argwhere(mask.any(axis=1)).reshape(-1)
    mask_col = np.nanargmax(mask[mask_row, :], axis=1)
    for row, col in zip(mask_row, mask_col):
        mask[row, col:] = True
    mask_undefined = mask * mask_aerosol0
    df.data['classifier'][mask_undefined] = i

# %%
fig, ax = plt.subplots(figsize=(6, 2))
p = ax.pcolormesh(df.data['time'], df.data['range'],
                  df.data['classifier'].T,
                  cmap=cmap, norm=norm)
ax.yaxis.set_major_formatter(hd.m_km_ticks())
ax.set_ylabel('Range [km, a.g.l]')
ax.set_xlabel('Time [UTC - hour]')
cbar = fig.colorbar(p, ax=ax, ticks=[5, 15, 25, 35, 45])
cbar.ax.set_yticklabels(['Background', 'Aerosol',
                         'Precipitation', 'Clouds', 'Undefined'])
fig.tight_layout()
fig.savefig(path + '/algorithm_attenuation.png', bbox_inches='tight')

# %%
if (df.data['classifier'] == 10).any():
    classifier = df.data['classifier'].ravel()
    time_dbscan = np.repeat(np.arange(df.data['time'].size),
                            df.data['beta_raw'].shape[1])
    height_dbscan = np.tile(np.arange(df.data['range'].size),
                            df.data['beta_raw'].shape[0])

    time_dbscan = time_dbscan[classifier == 10].reshape(-1, 1)
    height_dbscan = height_dbscan[classifier == 10].reshape(-1, 1)
    X = np.hstack([time_dbscan, height_dbscan])
    db = DBSCAN(eps=3, min_samples=25, n_jobs=-1).fit(X)

    v_dbscan = v_save[classifier == 10]
    range_dbscan = range_save[classifier == 10]

    v_dict = {}
    r_dict = {}
    for i in np.unique(db.labels_):
        v_dict[i] = np.nanmean(v_dbscan[db.labels_ == i])
        r_dict[i] = np.nanmin(range_dbscan[db.labels_ == i])

    lab = db.labels_.copy()
    for key, val in v_dict.items():
        if key == -1:
            lab[db.labels_ == key] = 40
        elif (val < -0.5):
            lab[db.labels_ == key] = 20
        elif r_dict[key] == min(df.data['range']):
            lab[db.labels_ == key] = 10
        elif (val > -0.2):
            lab[db.labels_ == key] = 10
        else:
            lab[db.labels_ == key] = 40

    df.data['classifier'][df.data['classifier'] == 10] = lab

# %%
fig, ax = plt.subplots(figsize=(6, 2))
p = ax.pcolormesh(df.data['time'], df.data['range'],
                  df.data['classifier'].T,
                  cmap=cmap, norm=norm)
ax.yaxis.set_major_formatter(hd.m_km_ticks())
ax.set_ylabel('Range [km, a.g.l]')
ax.set_xlabel('Time [UTC - hour]')
cbar = fig.colorbar(p, ax=ax, ticks=[5, 15, 25, 35, 45])
cbar.ax.set_yticklabels(['Background', 'Aerosol',
                         'Precipitation', 'Clouds', 'Undefined'])
fig.tight_layout()
fig.savefig(path + '/algorithm_aerosol_finetuned.png', bbox_inches='tight')

# %%
# Separate ground rain
if (df.data['classifier'] == 20).any():
    classifier = df.data['classifier'].ravel()
    time_dbscan = np.repeat(np.arange(df.data['time'].size),
                            df.data['beta_raw'].shape[1])
    height_dbscan = np.tile(np.arange(df.data['range'].size),
                            df.data['beta_raw'].shape[0])

    time_dbscan = time_dbscan[classifier == 20].reshape(-1, 1)
    height_dbscan = height_dbscan[classifier == 20].reshape(-1, 1)
    X = np.hstack([time_dbscan, height_dbscan])
    db = DBSCAN(eps=3, min_samples=1, n_jobs=-1).fit(X)

    range_dbscan = range_save[classifier == 20]

    r_dict = {}
    for i in np.unique(db.labels_):
        r_dict[i] = np.nanmin(range_dbscan[db.labels_ == i])

    lab = db.labels_.copy()
    for key, val in r_dict.items():
        if r_dict[key] == min(df.data['range']):
            lab[db.labels_ == key] = 20
        else:
            lab[db.labels_ == key] = 30

    df.data['classifier'][df.data['classifier'] == 20] = lab

# %%
fig, ax = plt.subplots(figsize=(6, 2))
p = ax.pcolormesh(df.data['time'], df.data['range'],
                  df.data['classifier'].T,
                  cmap=cmap, norm=norm)
ax.yaxis.set_major_formatter(hd.m_km_ticks())
ax.set_ylabel('Range [km, a.g.l]')
ax.set_xlabel('Time [UTC - hour]')
cbar = fig.colorbar(p, ax=ax, ticks=[5, 15, 25, 35, 45])
cbar.ax.set_yticklabels(['Background', 'Aerosol',
                         'Precipitation', 'Clouds', 'Undefined'])
fig.tight_layout()
fig.savefig(path + '/algorithm_ground_precipitation.png', bbox_inches='tight')

###############################################################
# %%
###############################################################

data = hd.getdata('F:/halo/46/depolarization')

# %%
# date = '20180415'
date = '20180611'
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

log_beta = np.log10(df.data['beta_raw'])
# Aerosol
aerosol = log_beta < -5.5

# Small size median filter to remove noise
aerosol_smoothed = median_filter(aerosol, size=11)
# Remove thin bridges, better for the clustering
aerosol_smoothed = median_filter(aerosol_smoothed, size=(15, 1))

df.data['classifier'][aerosol_smoothed] = 10

df.filter(variables=['beta_raw', 'v_raw', 'depo_adj'],
          ref='co_signal',
          threshold=1 + 3 * df.snr_sd)
log_beta = np.log10(df.data['beta_raw'])

range_save = np.tile(df.data['range'],
                     df.data['beta_raw'].shape[0])

time_save = np.repeat(df.data['time'],
                      df.data['beta_raw'].shape[1])
v_save = df.data['v_raw'].flatten()  # put here to avoid noisy values at 1sd snr

# Liquid
liquid = log_beta > -5.5

# maximum filter to increase the size of liquid region
liquid_max = maximum_filter(liquid, size=5)
# Median filter to remove background noise
liquid_smoothed = median_filter(liquid_max, size=13)

df.data['classifier'][liquid_smoothed] = 30

# updraft - indication of aerosol zone
updraft = df.data['v_raw'] > 1
updraft_smooth = median_filter(updraft, size=3)
updraft_max = maximum_filter(updraft_smooth, size=91)

# Fill the gap in aerosol zone
updraft_median = median_filter(updraft_max, size=31)

# precipitation < -1 (center of precipitation)
precipitation_1 = (log_beta > -7) & (df.data['v_raw'] < -1)

precipitation_1_median = median_filter(precipitation_1, size=9)

# Only select precipitation outside of aerosol zone
precipitation_1_ne = precipitation_1_median * ~updraft_median
precipitation_1_median_smooth = median_filter(precipitation_1_ne,
                                              size=3)
precipitation = precipitation_1_median_smooth

# precipitation < -0.5 (include all precipitation)
precipitation_1_low = (log_beta > -7) & (df.data['v_raw'] < -0.5)

# Avoid ebola infection surrounding updraft
# Useful to contain error during ebola precipitation
updraft_ebola = df.data['v_raw'] > 0.2
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
    if i == 20:
        mask = df.data['classifier'] == i
    else:
        mask = log_beta > -5
        mask = maximum_filter(mask, size=5)
        mask = median_filter(mask, size=13)
    mask_row = np.argwhere(mask.any(axis=1)).reshape(-1)
    mask_col = np.nanargmax(mask[mask_row, :], axis=1)
    for row, col in zip(mask_row, mask_col):
        mask[row, col:] = True
    mask_undefined = mask * mask_aerosol0
    df.data['classifier'][mask_undefined] = i

# %%
if (df.data['classifier'] == 10).any():
    classifier = df.data['classifier'].ravel()
    time_dbscan = np.repeat(np.arange(df.data['time'].size),
                            df.data['beta_raw'].shape[1])
    height_dbscan = np.tile(np.arange(df.data['range'].size),
                            df.data['beta_raw'].shape[0])

    time_dbscan = time_dbscan[classifier == 10].reshape(-1, 1)
    height_dbscan = height_dbscan[classifier == 10].reshape(-1, 1)
    X = np.hstack([time_dbscan, height_dbscan])
    db = DBSCAN(eps=3, min_samples=25, n_jobs=-1).fit(X)

    v_dbscan = v_save[classifier == 10]
    range_dbscan = range_save[classifier == 10]

    v_dict = {}
    r_dict = {}
    for i in np.unique(db.labels_):
        v_dict[i] = np.nanmean(v_dbscan[db.labels_ == i])
        r_dict[i] = np.nanmin(range_dbscan[db.labels_ == i])

    lab = db.labels_.copy()
    for key, val in v_dict.items():
        if key == -1:
            lab[db.labels_ == key] = 40
        elif (val < -0.5):
            lab[db.labels_ == key] = 20
        elif r_dict[key] == min(df.data['range']):
            lab[db.labels_ == key] = 10
        elif (val > -0.2):
            lab[db.labels_ == key] = 10
        else:
            lab[db.labels_ == key] = 40

    df.data['classifier'][df.data['classifier'] == 10] = lab

# %%
# Separate ground rain
if (df.data['classifier'] == 20).any():
    classifier = df.data['classifier'].ravel()
    time_dbscan = np.repeat(np.arange(df.data['time'].size),
                            df.data['beta_raw'].shape[1])
    height_dbscan = np.tile(np.arange(df.data['range'].size),
                            df.data['beta_raw'].shape[0])

    time_dbscan = time_dbscan[classifier == 20].reshape(-1, 1)
    height_dbscan = height_dbscan[classifier == 20].reshape(-1, 1)
    X = np.hstack([time_dbscan, height_dbscan])
    db = DBSCAN(eps=3, min_samples=1, n_jobs=-1).fit(X)

    range_dbscan = range_save[classifier == 20]

    r_dict = {}
    for i in np.unique(db.labels_):
        r_dict[i] = np.nanmin(range_dbscan[db.labels_ == i])

    lab = db.labels_.copy()
    for key, val in r_dict.items():
        if r_dict[key] == min(df.data['range']):
            lab[db.labels_ == key] = 20
        else:
            lab[db.labels_ == key] = 30

    df.data['classifier'][df.data['classifier'] == 20] = lab

# %%
cmap = mpl.colors.ListedColormap(
    ['white', '#2ca02c', 'blue', 'red', 'gray'])
boundaries = [0, 10, 20, 30, 40, 50]
norm = mpl.colors.BoundaryNorm(boundaries, cmap.N, clip=True)

# %%
fig, ax = plt.subplots(4, 1, figsize=(6, 8))
ax1, ax3, ax5, ax7 = ax.ravel()
p1 = ax1.pcolormesh(df.data['time'], df.data['range'],
                    np.log10(df.data['beta_raw']).T, cmap='jet', vmin=-8, vmax=-4)
p2 = ax3.pcolormesh(df.data['time'], df.data['range'],
                    df.data['v_raw'].T, cmap='jet', vmin=-2, vmax=2)
p3 = ax5.pcolormesh(df.data['time'], df.data['range'],
                    df.data['depo_adj'].T, cmap='jet', vmin=0, vmax=0.5)
p4 = ax7.pcolormesh(df.data['time'], df.data['range'],
                    df.data['classifier'].T,
                    cmap=cmap, norm=norm)
for ax in [ax1, ax3, ax5, ax7]:
    ax.yaxis.set_major_formatter(hd.m_km_ticks())
    ax.set_ylabel('Range [km, a.g.l]')

cbar = fig.colorbar(p1, ax=ax1)
cbar.ax.set_ylabel('Beta [' + df.units.get('beta_raw', None) + ']', rotation=90)
# cbar.ax.yaxis.set_label_position('left')
cbar = fig.colorbar(p2, ax=ax3)
cbar.ax.set_ylabel('Velocity [' + df.units.get('v_raw', None) + ']', rotation=90)
# cbar.ax.yaxis.set_label_position('left')
cbar = fig.colorbar(p3, ax=ax5)
cbar.ax.set_ylabel('Depolarization ratio')
# cbar.ax.yaxis.set_label_position('left')
cbar = fig.colorbar(p4, ax=ax7, ticks=[5, 15, 25, 35, 45])
cbar.ax.set_yticklabels(['Background', 'Aerosol',
                         'Precipitation', 'Clouds', 'Undefined'])
ax7.set_xlabel('Time [UTC - hour]')

fig.tight_layout()
fig.savefig(path + '/algorithm_' + df.filename +
            '.png', bbox_inches='tight')


##############################################
# %% depo cloud base hist, ts
##############################################

# Define csv directory path
depo_paths = [
    'F:\\halo\\32\\depolarization\\depo',
    'F:\\halo\\33\\depolarization\\depo',
    'F:\\halo\\34\\depo',
    'F:\\halo\\46\\depolarization\\depo',
    'F:\\halo\\53\\depolarization\\depo',
    'F:\\halo\\54\\depolarization\\depo',
    'F:\\halo\\146\\depolarization\\depo']

# Collect csv file in csv directory and subdirectory
result_list = []
for csv_path in depo_paths:
    data_list = [file
                 for path, subdir, files in os.walk(csv_path)
                 for file in glob.glob(os.path.join(path, '*.csv'))]
    result_list.extend(data_list)

# %%
depo = pd.concat([pd.read_csv(f) for f in result_list],
                 ignore_index=True)
depo = depo.astype({'year': int, 'month': int, 'day': int, 'systemID': int})
depo = depo.astype({'systemID': str})
# For right now, just take the date, ignore hh:mm:ss
depo['date'] = pd.to_datetime(depo[['year', 'month', 'day']])
depo.drop(['depo_1', 'co_signal1'], axis=1, inplace=True)
depo.loc[(depo['date'] > '2017-11-20') & (depo['systemID'] == '32'),
         'systemID'] = '32XR'

# %%
device_config_path = glob.glob(r'F:\halo\summary_device_config/*.csv')
device_config_df = []
for f in device_config_path:
    df = pd.read_csv(f)
    location_sysID = f.split('_')[-1].split('.')[0].split('-')
    df['location'] = location_sysID[0]
    df['systemID'] = location_sysID[1]
    df.rename(columns={'time': 'date'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    device_config_df.append(df)

# %%
device_config = pd.concat(device_config_df, ignore_index=True)
device_config.loc[(device_config['date'] > '2017-11-20') &
                  (device_config['systemID'] == '32'),
                  'systemID'] = '32XR'
device_config.loc[device_config['systemID'].isin(['32XR', '146']),
                  'prf'] = 10000
device_config['integration_time'] = \
    device_config['num_pulses_m1'] / device_config['prf']

# %%
result = depo.merge(device_config, on=['systemID', 'date'])
temp = result.groupby(['systemID', 'date']).mean()

# %%
depo['sys'] = depo['location'] + '-' + depo['systemID']

# %%
for key, group in depo.groupby('sys'):
    # Co-cross histogram
    if key == 'Uto-32XR':
        co_cross_data = group[['co_signal', 'cross_signal']].dropna()
        H, co_edges, cross_edges = np.histogram2d(
            co_cross_data['co_signal'] - 1,
            co_cross_data['cross_signal'] - 1,
            bins=500)
        X, Y = np.meshgrid(co_edges, cross_edges)
        fig5, ax = plt.subplots(figsize=(6, 4))
        p = ax.pcolormesh(X, Y, H.T, norm=LogNorm())
        ax.set_xlabel('co_SNR')
        ax.set_ylabel('cross_SNR')
        colorbar = fig5.colorbar(p, ax=ax)
        colorbar.ax.set_ylabel('Number of observations')
        ax.plot(co_cross_data['co_signal'] - 1,
                (co_cross_data['co_signal'] - 1) * 0.01,
                label=r'$\frac{cross\_SNR}{co\_SNR} = 0.01$',
                linewidth=0.5)
        ax.legend(loc='upper left')
        fig5.savefig(path + '/' + key + '_cross_vs_co.png',
                     bbox_inches='tight')

    # Histogram of depo
    temp = group['depo']
    if key == 'Kumpula-146':
        mask = (temp < 0.4) & (temp > -0.05)
    else:
        mask = (temp < 0.2) & (temp > -0.05)
    fig6, ax = plt.subplots(figsize=(6, 3))
    temp.loc[mask].hist(bins=50)
    ax.set_xlabel('Depolarization ratio')

    fig6.savefig(path + '/' + key + '_depo_hist.png',
                 bbox_inches='tight')

    fig7, ax = plt.subplots(figsize=(6, 3))
    group_ = group.groupby('date').depo
    ax.errorbar(group['date'].unique(), group_.mean(), yerr=group_.std(),
                ls='none', marker='.', linewidth=0.5, markersize=5)
    ax.tick_params(axis='x', labelrotation=45)
    ax.set_ylabel('Depolarization ratio')
    fig7.savefig(path + '/' + key + '_depo_scatter.png',
                 bbox_inches='tight')

    plt.close('all')

# %%
for key, group in depo.groupby('sys'):
    # Co-cross histogram
    if key == 'Uto-32XR':
        fig6, ax = plt.subplots(figsize=(6, 3))
        temp = group['depo'][group['co_signal'] < 7]
        mask = (temp < 0.2) & (temp > -0.05)
        temp.loc[mask].hist(bins=50, label='co_SNR < 6', alpha=0.8)
        temp = group['depo'][group['co_signal'] > 7]
        mask = (temp < 0.2) & (temp > -0.05)
        temp.loc[mask].hist(bins=50, label='co_SNR > 6', alpha=0.8)
        ax.legend()
        ax.set_xlim([-0.05, 0.1])
        ax.set_xlabel('Depolarization ratio')
        fig6.tight_layout()

        fig6.savefig(path + '/' + 'saturation_Uto_32_hist.png',
                     bbox_inches='tight')


##############################################
# %%
##############################################

# %%
missing_df = pd.DataFrame({})
for site in ['46', '54', '33', '53', '34', '32']:
    path_ = 'F:\\halo\\classifier2\\' + site + '\\'
    list_files = glob.glob(path_ + '*.csv')
    time_df = pd.DataFrame(
        {'date': [file.split('\\')[-1][:10] for
                  file in list_files if 'result' not in file],
         'location2': [file.split('\\')[-1].split('-')[3] for
                       file in list_files if 'result' not in file]})
    time_df['date'] = pd.to_datetime(time_df['date'])
    time_df['year'] = time_df['date'].dt.year
    time_df['month'] = time_df['date'].dt.month
    time_df_count = time_df.groupby(['year', 'month', 'location2']).count()
    time_df_count = time_df_count.reset_index().rename(columns={'date': 'count'})
    missing_df = missing_df.append(time_df_count, ignore_index=True)
missing_df.loc[missing_df['location2'] == 'Kuopio', 'location2'] = 'Hyytiala'
missing_df = missing_df.set_index(['year', 'month', 'location2'])
mux = pd.MultiIndex.from_product([missing_df.index.levels[0],
                                  missing_df.index.levels[1],
                                  missing_df.index.levels[2]],
                                 names=['year', 'month', 'location2'])
missing_df = missing_df.reindex(mux, fill_value=0).reset_index()
missing_df_all = missing_df.copy()
missing_df = missing_df[missing_df['count'] < 15]

# %%
my_cmap = copy.copy(matplotlib.cm.get_cmap('jet'))
my_cmap.set_under('w')
bin_depo = np.linspace(0, 0.5, 50)
bin_month = np.arange(0.5, 13, 1)
bin_time = np.arange(0, 25)

# %%
df = pd.DataFrame()
for site in ['46', '54', '33', '53', '32']:
    save_location = 'F:\\halo\\classifier2\\summary\\'
    df = df.append(pd.read_csv('F:\\halo\\classifier2\\' + site + '\\result.csv'),
                   ignore_index=True)

df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['depo'][df['depo'] > 1] = np.nan
df.loc[df['location'] == 'Kuopio-33', 'location'] = 'Hyytiala-33'
df[['location2', 'ID']] = df['location'].str.split('-', expand=True)

# %%
df[(df['location2'] == 'Uto') &
    (df['date'] >= '2019-12-06') &
    (df['date'] <= '2019-12-10')] = np.nan

# %%
df_miss = df.merge(missing_df_all, 'outer')
df_miss.dropna(inplace=True)
df_miss = df_miss[df_miss['count'] >= 15]
fig, ax = plt.subplots(figsize=(6, 4))
for k, grp in df_miss.groupby(['location2']):
    grp.resample('M', on='date')['depo'].median().plot(ax=ax, label=k)
ax.set_ylabel('Depolarization ratio')
fig.legend(ncol=4, loc='upper center')
ax.set_xlabel('')
ax.grid(which='major', axis='x')
fig.savefig(path + '/depo_median_month.png',
            bbox_inches='tight')

# %%
season = {'Oct-Apr': [10, 11, 12, 1, 2, 3, 4],
          'May': [5],
          'Jun': [6],
          'Jul-Sep': [7, 8, 9]}

depo_season = {}
for k, grp in df.groupby(['location2']):
    depo_season[k] = {}
    for sea in season:
        temp = grp.loc[grp['month'].isin(season[sea]), 'depo']
        # temp = temp[temp > 0]
        # temp = temp[temp < 0.5]
        mean_ = temp.mean()
        sd_ = temp.std()
        depo_season[k][sea] = (mean_, sd_)

# %%
depo_season

# %%
jitter = {}
for place, v in zip(['Uto', 'Hyytiala', 'Vehmasmaki', 'Sodankyla'],
                    np.linspace(-0.15, 0.15, 4)):
    jitter[place] = v
fig, ax = plt.subplots(figsize=(6, 4))
for k, grp in df.groupby('location2'):
    dep = grp.groupby(grp.date.dt.month)['depo']
    dep_mean = dep.mean()
    dep_std = dep.std()
    ax.errorbar(dep_mean.index + jitter[k], dep_mean, yerr=dep_std, label=k,
                fmt='--', elinewidth=1)
ax.set_ylabel('Depolarization ratio')
ticklabels = [datetime.date(1900, item, 1).strftime('%b') for item in np.arange(1, 13, 3)]
ax.set_xticks(np.arange(1, 13, 3))
ax.set_xticklabels(ticklabels)
ax.grid(axis='x', which='major', linewidth=0.5, c='silver')
fig.legend(ncol=4, loc='upper center')
fig.savefig(path + '/depo_mean_month_avg.png',
            bbox_inches='tight')

# %%

y = {2016: 0, 2017: 1, 2018: 2, 2019: 3}
cbar_max = {'Uto': 600, 'Hyytiala': 600,
            'Vehmasmaki': 400, 'Sodankyla': 400}
# cbar_max = {'Uto': 600, 'Hyytiala': 450,
#             'Vehmasmaki': 330, 'Sodankyla': 260}

X, Y = np.meshgrid(bin_month, bin_depo)
for k, grp in df.groupby(['location2']):
    fig, axes = plt.subplots(2, 2, figsize=(6, 4), sharex=True)
    axes = axes.flatten()
    if k == 'Sodankyla':
        fig.delaxes(axes[0])
        axes = axes[1:]
    for (key, g), ax in zip(grp.groupby(['year']), axes):
        H, month_edge, depo_edge = np.histogram2d(
            g['month'], g['depo'],
            bins=(bin_month, bin_depo)
        )
        miss = missing_df[(missing_df['year'] == key) &
                          (missing_df['location2'] == k)]['month']
        if len(miss.index) != 0:
            for miss_month in miss:
                H[miss_month-1, :] = np.nan
        p = ax.pcolormesh(X, Y, H.T, cmap=my_cmap,
                          vmin=0.1, vmax=cbar_max[k])
        fig.colorbar(p, ax=ax)
        ax.xaxis.set_ticks([4, 8, 12])
        ax.set_ylabel('Depolarization ratio')
        ax.set_title(int(key), weight='bold')
        fig.tight_layout()
        fig.savefig(path + '/' + k + '_depo_month.png',
                    bbox_inches='tight')


# %% Plot percent per month
X, Y = np.meshgrid(bin_month, bin_depo)
fig, axes = plt.subplots(2, 2, figsize=(6, 4),
                         sharex=True, sharey=True)

axes[1, 0].set_xlabel('Month')
axes[1, 1].set_xlabel('Month')
axes[0, 0].set_ylabel('Depolarization ratio')
axes[1, 0].set_ylabel('Depolarization ratio')

avg = {}
for k, grp in df.groupby(['location2']):
    for key, g in grp.groupby(['year']):
        H, month_edge, depo_edge = np.histogram2d(
            g['month'], g['depo'],
            bins=(bin_month, bin_depo)
        )
        miss = missing_df[(missing_df['year'] == key) &
                          (missing_df['location2'] == k)]['month']
        if len(miss.index) != 0:
            for miss_month in miss:
                H[miss_month-1, :] = np.nan

        if k not in avg:
            avg[k] = H[:, :, np.newaxis]
        else:
            avg[k] = np.append(avg[k], H[:, :, np.newaxis], axis=2)

for (key, val), ax in zip(avg.items(), axes.flatten()):
    val = np.nansum(val, axis=2)
    val = val/(val.sum(axis=1).reshape(-1, 1))
    p = ax.pcolormesh(X, Y, val.T,
                      cmap=my_cmap,
                      vmin=0.001, vmax=0.2)
    ax.xaxis.set_ticks([4, 8, 12])
    ax.set_title(key, weight='bold')
    fig.colorbar(p, ax=ax)
    fig.tight_layout()
fig.savefig(path + '/all_depo_month_avg.png',
            bbox_inches='tight')


# %%
X, Y = np.meshgrid(bin_time, bin_month)
fig, axes = plt.subplots(2, 2, figsize=(6, 4), sharex=True, sharey=True)
avg = {}
for k, grp in df.groupby(['location2']):
    for key, g in grp.groupby(['year']):
        dep_mean, time_edge, month_edge, _ = binned_statistic_2d(
            g['time'],
            g['month'],
            g['depo'],
            bins=[bin_time, bin_month],
            statistic=np.nanmean)
        dep_count, _, _, _ = binned_statistic_2d(
            g['time'],
            g['month'],
            g['depo'],
            bins=[bin_time, bin_month],
            statistic='count')
        dep_mean[dep_count < 25] = np.nan
        if k not in avg:
            avg[k] = dep_mean[:, :, np.newaxis]
        else:
            avg[k] = np.append(avg[k], dep_mean[:, :, np.newaxis], axis=2)

for (key, val), ax, i in zip(avg.items(), axes.flatten(),
                             np.arange(4)):
    p = ax.pcolormesh(X, Y, np.nanmean(val, axis=2).T,
                      cmap='jet',
                      vmin=1e-5, vmax=0.3)
    ax.set_title(key, weight='bold')
    cbar = fig.colorbar(p, ax=ax)
    if i in [1, 3]:
        cbar.ax.set_ylabel('Depolarization ratio')

axes[0, 0].set_ylabel('Month')
axes[1, 0].set_ylabel('Month')
axes[1, 0].set_xlabel('Time (hour)')
axes[1, 1].set_xlabel('Time (hour)')

fig.tight_layout()
fig.savefig(path + '/' + 'month_time.png',
            bbox_inches='tight')

# %%
list_weather = glob.glob('F:/weather/*.csv')
location_weather = {'hyytiala': 'Hyytiala', 'kuopio': 'Vehmasmaki',
                    'sodankyla': 'Sodankyla', 'uto': 'Uto'}
weather = pd.DataFrame()
for file in list_weather:
    if 'kumpula' in file:
        continue
    df_file = pd.read_csv(file)
    df_file['location2'] = location_weather[file.split('\\')[-1].split('_')[0]]
    weather = weather.append(df_file, ignore_index=True)

weather = weather.rename(columns={'Vuosi': 'year', 'Kk': 'month',
                                  'Pv': 'day', 'Klo': 'time',
                                  'Suhteellinen kosteus (%)': 'RH',
                                  'Ilman lämpötila (degC)': 'Temperature'})
weather[['year', 'month', 'day']] = weather[['year',
                                             'month', 'day']].astype(str)
weather['month'] = weather['month'].str.zfill(2)
weather['day'] = weather['day'].str.zfill(2)
weather['datetime'] = weather['year'] + weather['month'] + \
    weather['day'] + weather['time']
weather['datetime'] = pd.to_datetime(weather['datetime'], format='%Y%m%d%H:%M')

df['hour'] = np.floor(df['time'])
df['minute'] = (df['time'] % 1) * 60
df['second'] = 0
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute', 'second']])
df = df.drop(['year', 'month', 'day', 'hour', 'minute', 'second'], axis=1)


weather = weather.set_index('datetime').resample('0.5H').mean()
weather = weather.reset_index()
weather['datetime'] = weather['datetime'] + pd.Timedelta(minutes=15)

df = pd.merge(weather, df)

# %%
jitter = {'Uto': [-60, -2], 'Hyytiala': [-30, -1],
          'Vehmasmaki': [0, 0], 'Sodankyla': [30, 1]}
period_months = np.arange(1, 13).reshape(4, 3)
month_labs = [calendar.month_abbr[i] for i in np.arange(1, 13)]
month_labs = ['-'.join(month_labs[ii-3:ii]) for ii in [3, 6, 9, 12]]

# period_months = np.array([[7, 8, 9], [10, 11, 12, 1, 2, 3, 4], [5], [6]])
# month_labs = ['Jul-Sep', 'Oct-Apr', 'May', 'June']

fig, axes = plt.subplots(2, 2, figsize=(6, 5), sharey=True, sharex=True)
fig2, axes2 = plt.subplots(2, 2, figsize=(6, 5), sharey=True, sharex=True)

for (i, month), ax, ax2 in zip(enumerate(period_months),
                               axes.flatten(), axes2.flatten()):
    # for loc, grp in df[(df.datetime.dt.month >= month[0]) & (df.datetime.dt.month <= month[1])].groupby('location2'):
    for loc, grp in df[df.datetime.dt.month.isin(month)].groupby('location2'):
        grp_avg = grp.groupby('range')['depo'].mean()
        grp_count = grp.groupby('range')['depo'].count()
        grp_std = grp.groupby('range')['depo'].std()
        grp_avg = grp_avg[grp_count > 0.01 * sum(grp_count)]
        grp_std = grp_std[grp_count > 0.01 * sum(grp_count)]
        ax.errorbar(grp_avg.index + jitter[loc][0],
                    grp_avg, grp_std, label=loc,
                    fmt='.', elinewidth=1)
        ax.set_xlim([-50, 2500])
        grp_ground = grp[grp['range'] <= 300]
        y_grp = grp_ground.groupby(pd.cut(grp_ground['RH'], np.arange(0, 110, 10)))
        y_mean = y_grp['depo'].mean()
        y_std = y_grp['depo'].std()
        x = np.arange(5, 105, 10)
        ax2.errorbar(x + jitter[loc][1], y_mean, y_std,
                     label=loc, fmt='.', elinewidth=1)
        ax2.set_xlim([20, 100])
    if i in [2, 3]:
        ax.set_xlabel('Range [km, a.g.l]')
        ax.xaxis.set_major_formatter(hd.m_km_ticks())
        ax2.set_xlabel('Relative humidity')
    if i in [0, 2]:
        ax.set_ylabel('Depolarization ratio')
        ax2.set_ylabel('Depolarization ratio')
    ax.set_title(month_labs[i], weight='bold')
    ax2.set_title(month_labs[i], weight='bold')
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4)
handles, labels = ax2.get_legend_handles_labels()
fig2.legend(handles, labels, loc='upper center', ncol=4)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig2.tight_layout(rect=(0, 0, 1, 0.9))
# fig.savefig(path + '/depo_range.png', bbox_inches='tight')
# fig2.savefig(path + '/depo_RH.png', bbox_inches='tight')

# %%


def regress(data, yvar, xvars):
    Y = data[yvar]
    X = data[xvars]
    X['intercept'] = 1.
    result = sm.OLS(Y, X).fit()
    return [np.around(i, digit) for i, digit in zip([result.params[0], result.pvalues[0], result.rsquared],
                                                    [3, 5, 3])]


temp = df.copy()
temp['range'] = temp['range']/1000
temp.dropna(inplace=True)
for (i, month) in enumerate(period_months):
    print(month,
          temp[temp.datetime.dt.month.isin(month)].groupby('location2').apply(regress, 'depo', ['range']))


# %%
temp['RH'] = temp['RH']/10

for (i, month) in enumerate(period_months):
    print(month,
          temp[temp.datetime.dt.month.isin(month)].groupby('location2').apply(regress, 'depo', ['RH']))

# %%
jitter = {'Uto': [-60, -0.03], 'Hyytiala': [-30, -0.01],
          'Vehmasmaki': [0, 0.01], 'Sodankyla': [30, 0.03]}
period_months = np.arange(1, 13).reshape(4, 3)
month_labs = [calendar.month_abbr[i] for i in np.arange(1, 13)]
month_labs = ['-'.join(month_labs[ii-3:ii]) for ii in [3, 6, 9, 12]]
# period_months = np.array([[7, 8, 9], [10, 11, 12, 1, 2, 3, 4], [5], [6]])
# month_labs = ['Jul-Sep', 'Oct-Apr', 'May', 'June']

fig, axes = plt.subplots(2, 2, figsize=(6, 5), sharey=True, sharex=True)
fig2, axes2 = plt.subplots(2, 2, figsize=(6, 5), sharey=True, sharex=True)

for (i, month), ax, ax2 in zip(enumerate(period_months),
                               axes.flatten(), axes2.flatten()):
    # for loc, grp in df[(df.datetime.dt.month >= month[0]) & (df.datetime.dt.month <= month[1])].groupby('location2'):
    for loc, grp in df[df.datetime.dt.month.isin(month)].groupby('location2'):
        grp_grp = grp.groupby(pd.cut(grp['depo'], np.arange(0, 0.5, 0.05)))
        grp_avg = grp_grp['range'].mean()
        grp_count = grp_grp['range'].count()
        grp_std = grp_grp['range'].std()
        grp_avg = grp_avg[grp_count > 0.01 * sum(grp_count)]
        grp_std = grp_std[grp_count > 0.01 * sum(grp_count)]
        x = np.arange(0.025, 0.025 + 0.05*len(grp_avg), 0.05)
        ax.errorbar(grp_avg, x[:len(grp_avg)] + jitter[loc][1], xerr=grp_std, label=loc,
                    fmt='--', elinewidth=1)
        ax.set_xlim([-50, 2500])
        grp_ground = grp[grp['range'] <= 300]
        y_grp = grp_ground.groupby(pd.cut(grp_ground['depo'], np.arange(0, 0.5, 0.05)))
        y_mean = y_grp['RH'].mean()
        y_std = y_grp['RH'].std()
        x = np.arange(0, 0.45, 0.05) + 0.025
        ax2.errorbar(y_mean, x + jitter[loc][1], xerr=y_std,
                     label=loc, fmt='--', elinewidth=1)
        ax2.set_ylim([0, 0.5])
    if i in [2, 3]:
        ax.set_xlabel('Range [m, a.g.l]')
        ax2.set_xlabel('Relative humidity')
    if i in [0, 2]:
        ax.set_ylabel('Depolarization ratio')
        ax2.set_ylabel('Depolarization ratio')
    ax.set_title(month_labs[i], weight='bold')
    ax2.set_title(month_labs[i], weight='bold')
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4)
handles, labels = ax2.get_legend_handles_labels()
fig2.legend(handles, labels, loc='upper center', ncol=4)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig2.tight_layout(rect=(0, 0, 1, 0.9))
# fig.savefig(path + '/depo_range.png', bbox_inches='tight')
# fig2.savefig(path + '/depo_RH.png', bbox_inches='tight')

# %%
fig, ax = plt.subplots()
ax.errorbar(np.arange(9), y_mean)

##############################################
# %% snr cloud base
##############################################

# %%
# Specify data folder path
data_folder = r'F:\halo\32\depolarization'

# %%
# Get a list of all files in data folder
data = hd.getdata(data_folder)

# %%
# pick date of data
pick_date = '20160223'
data_indices = hd.getdata_date(data, pick_date, data_folder)
df = hd.halo_data(data[data_indices])
print(df.filename)
# Change masking missing values from -999 to NaN
df.unmask999()
# Remove bottom 3 height layers
df.filter_height()

df.filter(variables=['beta_raw', 'depo_raw'],
          ref='co_signal',
          threshold=1 + 3*df.snr_sd)

# %%
fig, ax = plt.subplots(1, 2, figsize=(6, 4), sharey=True)
time = (17.645 <= df.data['time']) & (df.data['time'] <= 17.646)
range_mask = df.data['range'] < 1000
snr = df.data['co_signal'][time, range_mask]
depo = df.data['depo_raw'][time, range_mask]
ax[1].scatter(snr.ravel(), df.data['range'][range_mask], c=snr)
ax[0].scatter(depo.ravel(), df.data['range'][range_mask], c=snr)
ax[0].set_ylabel('Range [m.a.g.l]')
ax[0].set_xlabel('Depolarization ratio')
ax[1].set_xlabel('co-SNR')
fig.tight_layout()
fig.savefig(path + '/cloud_base_SNR.png', bbox_inches='tight')

##############################################
# %% snr ts
##############################################

# %%
csv_path = r'F:\halo\32\depolarization\snr'
# Collect csv file in csv directory
data_list = glob.glob(csv_path + '/*_noise.csv')
# %%
noise = pd.concat([pd.read_csv(f) for f in data_list],
                  ignore_index=True)
noise = noise.astype({'year': int, 'month': int, 'day': int})
noise['time'] = pd.to_datetime(noise[['year', 'month', 'day']]).dt.date
name = noise['location'][0] + '-' + str(int(noise['systemID'][0]))

# %%
# Calculate metrics
groupby = noise.groupby('time')['noise']
sd = groupby.std()
mean = groupby.mean()

# %%
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(sd, '.', markersize=0.5)
ax.set_ylabel('Standard deviation of SNR')
ax.set_ylim([0, 0.0025])
ax.tick_params(axis='x', labelrotation=45)
fig.tight_layout()
fig.savefig(path + '/snr_ts.png', bbox_inches='tight')

##############################################
# %% snr ts
##############################################

data = hd.getdata('F:/halo/32/depolarization')

# %%
date = '20190405'
file = [file for file in data if date in file][0]
df = hd.halo_data(file)

df.filter_height()
df.unmask999()
df.depo_cross_adj()

# %%
with open('ref_XR2.npy', 'rb') as f:
    ref = np.load(f)
df.filter(variables=['beta_raw'],
          ref='co_signal',
          threshold=1 + 3*df.snr_sd)
# %%
log_beta = np.log10(df.data['beta_raw'])
log_beta2 = log_beta.copy()
log_beta2[:, :50] = log_beta2[:, :50] - ref

# %%
fig, ax = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
for ax_, beta in zip(ax.flatten(), [log_beta, log_beta2]):
    p = ax_.pcolormesh(df.data['time'], df.data['range'],
                       beta.T, cmap='jet', vmin=-8, vmax=-4)
    cbar = fig.colorbar(p, ax=ax_)
    cbar.ax.set_ylabel('Attenuated backscatter')
    ax_.yaxis.set_major_formatter(hd.m_km_ticks())
    ax_.set_ylabel('Range [km, a.g.l]')
ax_.set_xlabel('Time [UTC - hour]')
fig.tight_layout()
fig.savefig(path + '/XR_correction_' + df.filename +
            '.png', bbox_inches='tight')

# %%


def mk_groups(data):
    try:
        newdata = data.items()
    except:
        return

    thisgroup = []
    groups = []
    for key, value in newdata:
        newgroups = mk_groups(value)
        if newgroups is None:
            thisgroup.append((key, value))
        else:
            thisgroup.append((key, len(newgroups[-1])))
            if groups:
                groups = [g + n for n, g in zip(newgroups, groups)]
            else:
                groups = newgroups
    return [thisgroup] + groups


def add_line(ax, ypos, xpos):
    line = plt.Line2D(xpos, ypos,
                      transform=ax.transAxes, color='black')
    line.set_clip_on(False)
    ax.add_line(line)


def label_group_bar(ax, data):
    color_err = ['white', '#2ca02c', 'blue', 'red', 'gray']
    groups = mk_groups(data)
    xy = groups.pop()
    y, xx = zip(*xy)
    x = [i[0] for i in xx]
    xsd = [i[1] for i in xx]
    ly = len(y)
    yticks = range(1, ly + 1)

    ax.errorbar(x, yticks, xerr=xsd, fmt='none',
                ecolor=color_error_bar)
    ax.scatter(x[:-2], yticks[:-2], c=color_error_bar[:-2], s=30)
    ax.set_yticks(yticks)
    ax.set_yticklabels(y)
    ax.set_ylim(.5, ly + .5)
    ax.set_xlim(0, .4)
    ax.xaxis.grid(True)
    # ax.yaxis.grid(True)

    scale = 1. / ly
    for pos in range(ly + 1):
        add_line(ax, [pos * scale, pos * scale], [-.12, 0])
    ypos = -.2
    level = 2
    while groups:
        group = groups.pop()
        pos = 0
        for label, rpos in group:
            lxpos = (pos + .5 * rpos) * scale
            if level == 1:
                ax.text(ypos-.15, lxpos, label,
                        ha='center', transform=ax.transAxes, va='center',
                        weight='bold', size=9)
                add_line(ax, [pos * scale, pos * scale], [ypos - .2, ypos+.1])
            else:
                ax.text(ypos-.02, lxpos, label, ha='center',
                        transform=ax.transAxes, va='center')
                add_line(ax, [pos * scale, pos * scale], [ypos - .1, ypos+.1])
            pos += rpos
        add_line(ax, [pos * scale, pos * scale], [ypos - .2, ypos+.1])
        ypos -= .1
        level -= 1


my_cmap = plt.cm.get_cmap('jet')
wave_color = {'355nm': my_cmap(0.99), '532nm': my_cmap(0.75), '710nm': my_cmap(0.5),
              '1064nm': my_cmap(0.25), '1565nm': my_cmap(0)}

data = {'Vakkari \net al. (2020)':
        {'Saharan dust':
         {'355nm': (0.19, 0.008),
          '532nm': (0.23, 0.008),
          '1565nm': (0.29, 0.008)},
         'Egyption dust':
         {'355nm': (0.36, 0.01),
          '532nm': (0.34, 0.002),
          '1565nm': (0.30, 0.005)},
             'Turkish dust':
         {'355nm': (0.31, 0.006),
          '532nm': (0.33, 0.005),
          '1565nm': (0.32, 0.008)}
         },
        'Haarig \n et al. (2017)':
            {'Saharan dust':
             {'355nm': (0.252, 0.03),
              '532nm': (0.28, 0.02),
              '1064nm': (0.225, 0.022)}
             },
        'Burton \n et al. (2015)':
            {'Saharan dust':
             {'355nm': (0.246, 0.0018),
              '532nm': (0.304, 0.005),
              '710nm': (0.270, 0.005)},
             'Mexican dust':
             {'355nm': (0.243, 0.046),
              '532nm': (0.373, 0.014),
              '710nm': (0.383, 0.006)},
             },
        'Freudenthaler \n et al. (2009)':
            {'Saharan dust':
             {'532nm': (0.31, 0.03),
              '1064nm': (0.27, 0.04)}
             },
        'Groß \n et al. (2011)':
            {'marine':
             {'355nm': (0.02, 0.01),
              '532nm': (0.02, 0.02)}
             },
        'Vakkari \n et al. (2020)':
            {'polluted \n marine':
             {'355nm': (0.03, 0.01),
              '532nm': (0.015, 0.002),
              '1565nm': (0.009, 0.003)}
             },
        'Vakkari \n et al.(2020)':
            {'spruce + birch \n pollen':
             {'532nm': (0.236, 0.009),
              '1565nm': (0.269, 0.005)}
             },
        'Bohlmann \n et al. (2019)':
            {'birch pollen':
             {'532nm': (0.1, 0.06)},
             'spruce + birch \n pollen':
             {'532nm': (0.26, 0.07)}
             },
        'Groß \n et al. (2012)':
            {'volcanic ash':
             {'355nm': (0.365, 0.015),
              '532nm': (0.365, 0.015)}
             }
        }

color_error_bar = []
for k1, i1 in data.items():
    for k2, i2 in i1.items():
        for k3, i3 in i2.items():
            color_error_bar.append(wave_color[k3])


fig, ax = plt.subplots(figsize=(9, 13))
label_group_bar(ax, data)
ax.xaxis.set_tick_params(labeltop='on')

legend_elements = [Line2D([], [], label='355nm', color=wave_color['355nm'], linewidth=1.5, marker='o',
                          markerfacecolor=wave_color['355nm'], markeredgewidth=0, markersize=7),
                   Line2D([], [], label='532nm', color=wave_color['532nm'], linewidth=1.5, marker='o',
                          markerfacecolor=wave_color['532nm'], markeredgewidth=0, markersize=7),
                   Line2D([], [], label='710nm', color=wave_color['710nm'], linewidth=1.5, marker='o',
                          markerfacecolor=wave_color['710nm'], markeredgewidth=0, markersize=7),
                   Line2D([], [], label='1064nm', color=wave_color['1064nm'], linewidth=1.5, marker='o',
                          markerfacecolor=wave_color['1064nm'], markeredgewidth=0, markersize=7),
                   Line2D([], [], label='1565nm', color=wave_color['1565nm'], linewidth=1.5, marker='o',
                          markerfacecolor=wave_color['1565nm'], markeredgewidth=0, markersize=7)]
fig.legend(handles=legend_elements, ncol=5, loc='upper center')
add_line(ax, [1/31*20, 1/31*20], [0, 1])
add_line(ax, [1/31*25, 1/31*25], [0, 1])
add_line(ax, [1/31*29, 1/31*29], [0, 1])
fig.subplots_adjust(left=0.2)
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig(path + '/ref_depo.png', bbox_inches='tight')

# %%
data = {'Vakkari \n et al. (2020)':
        {'Saharan dust':
         {'355nm': (0.19, 0.008),
          '532nm': (0.23, 0.008),
          '1565nm': (0.29, 0.008)},
         'Egyption dust':
         {'355nm': (0.36, 0.01),
          '532nm': (0.34, 0.002),
          '1565nm': (0.30, 0.005)},
             'Turkish dust':
         {'355nm': (0.31, 0.006),
          '532nm': (0.33, 0.005),
          '1565nm': (0.32, 0.008)},
         'polluted \n marine':
         {'355nm': (0.03, 0.01),
          '532nm': (0.015, 0.002),
          '1565nm': (0.009, 0.003)},
         'spruce + birch \n pollen':
         {'532nm': (0.236, 0.009),
          '1565nm': (0.269, 0.005)}
         },
        'Haarig \n et al. (2017)':
            {'Saharan dust':
             {'355nm': (0.252, 0.03),
              '532nm': (0.28, 0.02),
              '1064nm': (0.225, 0.022)}
             },
        'Haarig \n et al. (2018)':
            {'Stratosphere smoke':
             {'355nm': (0.224, 0.015),
              '532nm': (0.184, 0.006),
              '1064nm': (0.043, 0.007)},
             'Troposphere smoke':
             {'355nm': (0.021, 0.040),
              '532nm': (0.029, 0.015),
              '1064nm': (0.009, 0.008)}
             },
        'Pereia \n et al. (2014)':
            {'Troposphere smoke':
             {'532nm': (0.05, 0.01)}
             },
        'Baars \n et al. (2013)':
            {'Troposphere smoke':
             {'355nm': (0.025, 0.01)}
             },
        'Burton \n et al. (2015)':
            {'Saharan dust':
             {'355nm': (0.246, 0.0018),
              '532nm': (0.304, 0.005),
              '710nm': (0.270, 0.005)},
             'Mexican dust':
             {'355nm': (0.243, 0.046),
              '532nm': (0.373, 0.014),
              '710nm': (0.383, 0.006)},
             'Stratosphere smoke':
                {'355nm': (0.203, 0.036),
                 '532nm': (0.093, 0.015),
                 '1064nm': (0.018, 0.002)}},
        'Freudenthaler \n et al. (2009)':
            {'Saharan dust':
             {'532nm': (0.31, 0.03),
              '1064nm': (0.27, 0.04)}
             },
        'Groß \n et al. (2011)':
            {'marine':
             {'355nm': (0.02, 0.01),
              '532nm': (0.02, 0.02)}
             },
        'Bohlmann \n et al. (2019)':
            {'birch pollen':
             {'532nm': (0.1, 0.06)},
             'spruce + birch \n pollen':
             {'532nm': (0.26, 0.07)}
             },
        'Groß \n et al. (2012)':
            {'volcanic ash':
             {'355nm': (0.365, 0.015),
              '532nm': (0.365, 0.015)}
             }
        }

# %%
dust = []
pollen = []
marine = []
ash = []
smoke = []
for k1, v1 in data.items():
    for k2, v2 in v1.items():
        if 'dust' in k2:
            dust.append({k1 + ' - ' + k2: v2})
        elif 'pollen' in k2:
            pollen.append({k1 + ' - ' + k2: v2})
        elif 'marine' in k2:
            marine.append({k1 + ' - ' + k2: v2})
        elif 'ash' in k2:
            ash.append({k1 + ' - ' + k2: v2})
        elif 'smoke' in k2:
            smoke.append({k1 + ' - ' + k2: v2})
        else:
            print('nooooooooo')

# %%
fig, axes = plt.subplots(6, 1, figsize=(5, 15))
axes = axes.flatten()
for type, ax in zip([dust, pollen, marine, ash, smoke], axes):
    for i in type:
        x = []
        y = []
        yerr = []
        for author, vi in i.items():
            for wa, vii in vi.items():
                x.append(int(wa[:-2]))
                y.append(vii[0])
                yerr.append(vii[1])
            author = author.replace(' \n', '')
            ax.errorbar(np.array(x), y, yerr=yerr, fmt='.-',
                        elinewidth=1, label=author)
    ax.legend()
    ax.tick_params(axis='x', bottom=False)
    ax.set_ylabel('Depolarization ratio')
    ax.set_ylim([-0.07, 0.4])
    ax.set_xlim([300, 1600])
    ax.set_xticks([355, 532, 710, 1064, 1565])
    ax.legend(bbox_to_anchor=(1.01, 1.05), loc="upper left")

grp = df.groupby('location2')['depo']
x = 1565 + np.array([-20, -7, 7, 20])
for xx, mean, std, (lab, v) in zip(x, grp.mean(), grp.std(), grp):
    axes[-1].errorbar(xx, mean, std, elinewidth=1, fmt='.-', label=lab)
axes[-1].set_xticks([355, 532, 710, 1064, 1565])
axes[-1].set_xlabel('Wavelength (nm)')
axes[-1].legend(bbox_to_anchor=(1.01, 1.05), loc="upper left")
axes[-1].set_ylabel('Depolarization ratio')
axes[-2].set_ylabel('Depolarization ratio')
axes[-1].set_ylim([-0.07, 0.4])
axes[-1].set_xlim([300, 1600])
axes[-1].tick_params(axis='x', bottom=False)
fig.subplots_adjust(hspace=0.3)
fig.savefig(path + '/depo_vs_wave.png', bbox_inches='tight')

# %%
auth = {i: ma for i, ma in zip(data,
                               ['.', 'v', 's',
                                'p', 'H', '*',
                                'D'])}
auth

color_ = [co for co in wave_color.values()]
color_

# %%
fig, ax = plt.subplots(figsize=(6, 7))
# wave = {'355nm': 355, '532nm': 1, '710nm': 2, '1064nm': 3, '1565nm': 4}
for type, colo in zip([dust, pollen, marine, ash], [co for co in wave_color.values()]):
    for i in type:
        x = []
        y = []
        yerr = []
        for author, vi in i.items():
            for wa, vii in vi.items():
                x.append(int(wa[:-2]))
                y.append(vii[0])
                yerr.append(vii[1])
            ran = np.random.randint(-20, 20)
            ax.errorbar(np.array(x)+ran, y, yerr=yerr, color=colo, fmt='--', elinewidth=1)
            ax.scatter(np.array(x)+ran, y, color=colo, marker=auth[author])

grp = df.groupby('location2')['depo']
x = 1565 + np.array([-20, -7, 7, 20])
for xx, mean, std, marker in zip(x, grp.mean(),
                                 grp.std(), [r'$\alpha$', r'$\beta$',
                                             r'$\gamma$', r'$\delta$']):
    ax.errorbar(xx, mean, std, marker=marker, elinewidth=1, color='darkblue')
ax.set_xticks([355, 532, 710, 1064, 1565])
ax.set_ylabel('Depolarization ratio')
ax.set_xlabel('Wavelength (nm)')
ax.tick_params(axis='x', bottom=False)

auth_ = {'Vakkari et al. (2020) - dust': ['.', color_[0]],
         'Haarig et al. (2017) - dust': ['v', color_[0]],
         'Burton et al. (2015) - dust': ['s', color_[0]],
         'Freudenthaler et al. (2009) - dust': ['p', color_[0]],
         'Groß et al. (2011) - marine': ['H', color_[2]],
         'Vakkari et al. (2020) - marine': ['.', color_[2]],
         'Vakkari et al. (2020) - pollen': ['.', color_[1]],
         'Bohlmann et al. (2019) - pollen': ['*', color_[1]],
         'Groß et al. (2012) - volcanic ash': ['D', color_[3]]}

for (gr, v), mark in zip(grp, [r'$\alpha$', r'$\beta$',
                               r'$\gamma$', r'$\delta$']):
    auth_[gr] = [mark, 'darkblue']

legend_elements = [Line2D([], [], label=k, color=i[1], linewidth=1.5, marker=i[0],
                          markerfacecolor=i[1], markeredgewidth=0, markersize=7) for k, i in auth_.items()]
fig.legend(handles=legend_elements, ncol=2, loc='lower center')
fig.tight_layout(rect=(0, 0.25, 1, 1))
fig.savefig(path + '/depo_vs_wave.png', bbox_inches='tight')

# %%

depo_season
depo_season2 = {}
depo_season2['Uto'] = depo_season['Uto']
depo_season2['Hyytiala'] = depo_season['Hyytiala']
depo_season2['Vehmasmaki'] = depo_season['Vehmasmaki']
depo_season2['Sodankyla'] = depo_season['Sodankyla']
depo_season = depo_season2
for place in depo_season:
    for months, v in depo_season[place].items():
        depo_season[place][months] = {'1565nm': v}


# %%


def label_group_bar(ax, data):
    color_err = ['white', '#2ca02c', 'blue', 'red', 'gray']
    groups = mk_groups(data)
    xy = groups.pop()
    y, xx = zip(*xy)
    x = [i[0] for i in xx]
    xsd = [i[1] for i in xx]
    ly = len(y)
    yticks = range(1, ly + 1)

    ax.errorbar(x, yticks, xerr=xsd, fmt='none',
                ecolor=wave_color['1565nm'])
    ax.scatter(x, yticks, color=wave_color['1565nm'], s=30)
    ax.set_yticks(yticks)
    ax.set_yticklabels(y)
    ax.set_ylim(.5, ly + .5)
    ax.set_xlim(0, .4)
    ax.xaxis.grid(True)
    # ax.yaxis.grid(True)

    scale = 1. / ly
    for pos in range(ly + 1):
        add_line(ax, [pos * scale, pos * scale], [-.12, 0])
    ypos = -.2
    level = 2
    while groups:
        group = groups.pop()
        pos = 0
        for label, rpos in group:
            lxpos = (pos + .5 * rpos) * scale
            if level == 1:
                ax.text(ypos-.15, lxpos, label,
                        ha='center', transform=ax.transAxes, va='center',
                        weight='bold', size=9)
                add_line(ax, [pos * scale, pos * scale], [ypos - .2, ypos+.1])
            else:
                ax.text(ypos-.02, lxpos, label, ha='center',
                        transform=ax.transAxes, va='center')
                add_line(ax, [pos * scale, pos * scale], [ypos - .1, ypos+.1])
            pos += rpos
        add_line(ax, [pos * scale, pos * scale], [ypos - .2, ypos+.1])
        ypos -= .1
        level -= 1


fig, ax = plt.subplots(figsize=(9, 13))
label_group_bar(ax, depo_season)
ax.xaxis.set_tick_params(labeltop='on')
legend_elements = [Line2D([], [], label='1565nm', color=wave_color['1565nm'], linewidth=1.5, marker='o',
                          markerfacecolor=wave_color['1565nm'], markeredgewidth=0, markersize=7)]
fig.legend(handles=legend_elements, loc='upper center')
add_line(ax, [1/16*4, 1/16*4], [0, 1])
add_line(ax, [1/16*8, 1/16*8], [0, 1])
add_line(ax, [1/16*12, 1/16*12], [0, 1])
fig.subplots_adjust(left=0.2)
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig(path + '/ref_depo_result.png', bbox_inches='tight')

# %%

data = hd.getdata('F:/halo/46/depolarization')

# %%
date = '20180611'
file = [file for file in data if date in file][0]
df = hd.halo_data(file)

df.filter_height()
df.unmask999()
df.depo_cross_adj()
df.filter(variables=['beta_raw', 'v_raw', 'cross_signal', 'depo_raw'],
          ref='co_signal', threshold=1+3*df.snr_sd)
df.data['co_signal'][df.data['co_signal'] < 0.995] = np.nan

# %%
fig, axes = plt.subplots(figsize=(6, 2))
p = axes.pcolormesh(df.data['time'], df.data['range'],
                    np.log10(df.data['beta_raw']).T, cmap='jet',
                    vmin=-8, vmax=-4)
axes.set_xlim([0, 24])
cbar = fig.colorbar(p, ax=axes, fraction=0.05)
cbar.ax.set_ylabel('Beta [' + df.units.get('beta_raw', None) + ']', rotation=90)
axes.set_ylabel('Range [km, a.g.l]')
axes.yaxis.set_major_formatter(hd.m_km_ticks())
axes.set_xlabel('Time [UTC - hour]')
fig.tight_layout()
fig.savefig(path + '/raw_beta.png', bbox_inches='tight')

# %%
fig, axes = plt.subplots(figsize=(6, 2))
p = axes.pcolormesh(df.data['time'], df.data['range'],
                    df.data['v_raw'].T, cmap='jet',
                    vmin=-2, vmax=2)
axes.set_xlim([0, 24])
cbar = fig.colorbar(p, ax=axes, fraction=0.05)
cbar.ax.set_ylabel('Velocity [' + df.units.get('v_raw', None) + ']', rotation=90)
axes.set_ylabel('Range [km, a.g.l]')
axes.yaxis.set_major_formatter(hd.m_km_ticks())
axes.set_xlabel('Time [UTC - hour]')
fig.tight_layout()
fig.savefig(path + '/raw_v.png', bbox_inches='tight')

# %%
list_weather = glob.glob('F:/weather/*.csv')
location_weather = {'hyytiala': 'Hyytiala', 'kuopio': 'Vehmasmaki',
                    'sodankyla': 'Sodankyla', 'uto': 'Uto'}
weather = pd.DataFrame()
for file in list_weather:
    if 'kumpula' in file:
        continue
    df_file = pd.read_csv(file)
    df_file['location2'] = location_weather[file.split('\\')[-1].split('_')[0]]
    weather = weather.append(df_file, ignore_index=True)

weather = weather.rename(columns={'Vuosi': 'year', 'Kk': 'month',
                                  'Pv': 'day', 'Klo': 'time',
                                  'Suhteellinen kosteus (%)': 'RH',
                                  'Ilman lämpötila (degC)': 'Temperature'})
weather[['year', 'month', 'day']] = weather[['year',
                                             'month', 'day']].astype(str)
weather['month'] = weather['month'].str.zfill(2)
weather['day'] = weather['day'].str.zfill(2)
weather['datetime'] = weather['year'] + weather['month'] + \
    weather['day'] + weather['time']
weather['datetime'] = pd.to_datetime(weather['datetime'], format='%Y%m%d%H:%M')
weather['date_plot'] = weather['datetime'].dt.strftime('%m-%d')
weather = weather.set_index('datetime')

# %%
fig, axes = plt.subplots(4, 1, figsize=(6, 8), sharex=True, sharey=True)
for (k, g), ax in zip(weather.groupby('location2'), axes.flatten()):
    ax.plot(g.resample('MS').mean()['Temperature'])
    ax.grid()
    ax.set_title(k)

fig.subplots_adjust(hspace=0.3)

# %%
hyy = weather.loc[weather['location2'] == 'Hyytiala']

# %%
hyy.groupby(['year', 'month'])['Temperature'].mean().plot()

# %%
myFmt = mdates.DateFormatter('%m')

fig, ax = plt.subplots()
for (k, g), _ in zip(hyy.groupby('year'), np.arange(4)):
    ax.plot(g['date_plot'], g.Temperature, label=k)
ax.legend()
ax.xaxis.set_major_formatter(myFmt)

# %%
fig, axes = plt.subplots(2, 2, figsize=(7, 5),
                         sharex=True, sharey=True)
for (k, g), ax in zip(weather.groupby('location2'), axes.flatten()):
    pv = pd.pivot_table(g, index=g.index.month, columns=g.index.year,
                        values='Temperature', aggfunc='mean')
    pv = pv.iloc[:, :-1]
    pv.plot(ax=ax, legend=None)
    ax.set_title(k)
    ax.set_xlabel('Month')
    ax.xaxis.set_ticks_position('none')
    ax.set_xticks([1, 3, 5, 7, 9, 11])
    ax.grid()
    ax.set_ylabel('Temperature')
axes[0, 1].legend(title='Year', bbox_to_anchor=(1.01, 1.05), loc="upper left")
fig.tight_layout()
fig.savefig(path + '/temperature.png', bbox_inches='tight')
