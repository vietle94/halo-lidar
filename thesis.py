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
y = {2016: 0, 2017: 1, 2018: 2, 2019: 3}
cbar_max = {'Uto': 600, 'Hyytiala': 600,
            'Vehmasmaki': 400, 'Sodankyla': 400}
# cbar_max = {'Uto': 600, 'Hyytiala': 450,
#             'Vehmasmaki': 330, 'Sodankyla': 260}

X, Y = np.meshgrid(bin_month, bin_depo)
for k, grp in df.groupby(['location2']):
    avg = {}

    if k == 'Sodankyla':
        fig, axes = plt.subplots(2, 2, figsize=(6, 4), sharex=True)
        axes = axes.flatten()
        axes[2].set_xlabel('Month')
    else:
        fig = plt.figure(figsize=(6, 6))
        gs = fig.add_gridspec(3, 4)
        ax1 = fig.add_subplot(gs[0, :2])
        ax2 = fig.add_subplot(gs[0, 2:], sharey=ax1)
        ax1.axes.xaxis.set_ticklabels([])
        ax2.axes.xaxis.set_ticklabels([])
        ax3 = fig.add_subplot(gs[1, :2], sharey=ax1)
        ax3.set_xlabel('Month')
        ax4 = fig.add_subplot(gs[1, 2:])
        ax4.set_xlabel('Month')
        ax5 = fig.add_subplot(gs[2, 1:3])
        axes = [ax1, ax2, ax3, ax4, ax5]
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

        if k not in avg:
            avg[k] = H[:, :, np.newaxis]
        else:
            avg[k] = np.append(avg[k], H[:, :, np.newaxis], axis=2)

    for key, val in avg.items():

        p = axes[-1].pcolormesh(X, Y, np.nanmean(val, axis=2).T,
                                cmap=my_cmap,
                                vmin=0.1, vmax=cbar_max[key])
        axes[-1].xaxis.set_ticks([4, 8, 12])
        axes[-1].set_ylabel('Depolarization ratio')
        axes[-1].set_xlabel('Month')
        axes[-1].set_title('Averaged', weight='bold')
        fig.colorbar(p, ax=axes[-1])
        fig.tight_layout()
        fig.savefig(path + '/' + k + '_depo_month.png',
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


fig, axes = plt.subplots(2, 2, figsize=(6, 6), sharey=True, sharex=True)
fig2, axes2 = plt.subplots(2, 2, figsize=(6, 6), sharey=True, sharex=True)

for (i, month), ax, ax2 in zip(enumerate(period_months),
                               axes.flatten(), axes2.flatten()):
    for loc, grp in df[(df.datetime.dt.month >= month[0]) & (df.datetime.dt.month <= month[1])].groupby('location2'):
        grp_avg = grp.groupby('range')['depo'].mean()
        grp_count = grp.groupby('range')['depo'].count()
        grp_std = grp.groupby('range')['depo'].std()
        grp_avg = grp_avg[grp_count > 0.01 * sum(grp_count)]
        grp_std = grp_std[grp_count > 0.01 * sum(grp_count)]
        ax.errorbar(grp_avg.index + jitter[loc][0],
                    grp_avg, grp_std, label=loc, fmt='.')
        ax.set_xlim([-50, 2500])
        grp_ground = grp[grp['range'] <= 300]
        y_grp = grp_ground.groupby(pd.cut(grp_ground['RH'], np.arange(0, 110, 10)))
        y_mean = y_grp['depo'].mean()
        y_std = y_grp['depo'].std()
        x = np.arange(5, 105, 10)
        ax2.errorbar(x + jitter[loc][1], y_mean, y_std,
                     label=loc, fmt='.')
        ax2.set_xlim([20, 100])
    if i in [2, 3]:
        ax.set_xlabel('Range [km, a.g.l]')
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
fig.savefig(path + '/depo_range.png', bbox_inches='tight')
fig2.savefig(path + '/depo_RH.png', bbox_inches='tight')

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
