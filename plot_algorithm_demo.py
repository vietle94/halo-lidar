import json
from sklearn.metrics import r2_score
import dask.array as da
import matplotlib.cm as cm
import calendar
from scipy.stats import binned_statistic_2d
import os
from scipy.ndimage import uniform_filter
from sklearn.cluster import DBSCAN
import matplotlib.colors as colors
from scipy.ndimage import maximum_filter
from scipy.ndimage import median_filter
import matplotlib
import copy
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import datetime
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from sklearn.mixture import GaussianMixture
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import glob
import seaborn as sns
import numpy as np
from pathlib import Path
import matplotlib.dates as dates
import halo_data as hd
from matplotlib.colors import LogNorm
import xarray as xr
import string
import scipy.stats as stats
from netCDF4 import Dataset
import pywt
%matplotlib qt

# %%
path = r'F:\halo\paper\figures\algorithm/'
data = hd.getdata('F:/halo/46/depolarization')
date = '20180812'
file = [file for file in data if date in file][0]
df = hd.halo_data(file)

df.filter_height()
df.unmask999()
df.depo_cross_adj()

df.filter(variables=['beta_raw', 'v_raw'],
          ref='co_signal',
          threshold=1 + 3*df.snr_sd)

m_ = ((df.data['time'] > 5.70) &
      (df.data['time'] < 5.85)) | ((df.data['time'] > 6.20) &
                                   (df.data['time'] < 6.6))
m_ = m_ | ((df.data['time'] > 17.7) &
           (df.data['time'] < 17.9)) | ((df.data['time'] > 7.02) &
                                        (df.data['time'] < 7.1))

temp_co = df.data['co_signal'].copy()
lol = np.isnan(np.log10(df.data['beta_raw']))
lol[~m_, :] = False
temp_co[lol] = 1

fig, axes = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(6, 4.5))
p = axes[0].pcolormesh(df.data['time'], df.data['range'],
                       np.log10(df.data['beta_raw']).T, cmap='jet',
                       vmin=-8, vmax=-4)
axes[1].yaxis.set_major_formatter(hd.m_km_ticks())
axes[0].set_yticks([0, 4000, 8000])
axes[0].set_xlim([0, 24])
cbar = fig.colorbar(p, ax=axes[0], fraction=0.05)
cbar.ax.set_ylabel(r'$\beta\quad[Mm^{-1}]$', rotation=90)
cbar.ax.set_title(r'$1e$', size=10)
axes[0].set_ylabel('Height a.g.l [km]')
axes[-1].set_xlabel('Time UTC')

p = axes[1].pcolormesh(df.data['time'], df.data['range'],
                       df.data['v_raw'].T, cmap='jet',
                       vmin=-2, vmax=2)
cbar = fig.colorbar(p, ax=axes[1], fraction=0.05)
cbar.ax.set_ylabel('w [' + df.units.get('v_raw', None) + ']', rotation=90)
# axes[1].set_ylabel('Height a.g.l [km]')
axes[1].set_ylabel('Height a.g.l [km]')

p = axes[2].pcolormesh(df.data['time'], df.data['range'],
                       temp_co.T - 1, cmap='jet',
                       vmin=0.995 - 1, vmax=1.005 - 1)
cbar = fig.colorbar(p, ax=axes[2], fraction=0.05)
cbar.ax.set_ylabel('$SNR_{co}$', rotation=90)
# axes[2].set_ylabel('Height a.g.l [km]')
axes[2].set_ylabel('Height a.g.l [km]')
axes[2].set_xticks([0, 6, 12, 18, 24])
axes[2].set_xticklabels(['00:00', '06:00', '12:00', '18:00', '24:00'])
axes[0].set_ylim(bottom=0)
for n, ax in enumerate(axes.flatten()):
    ax.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
            transform=ax.transAxes, size=12)
fig.tight_layout()

fig.savefig(path + 'processed_data.png',
            bbox_inches='tight')

####################################################
# %%
####################################################
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
class1 = df.data['classifier'].copy()
class1[aerosol] = 10
# Small size median filter to remove noise
aerosol_smoothed = median_filter(aerosol, size=11)
# Remove thin bridges, better for the clustering
aerosol_smoothed = median_filter(aerosol_smoothed, size=(15, 1))

# %%
df.data['classifier'][aerosol_smoothed] = 10
class2 = df.data['classifier'].copy()
# %%
cmap = mpl.colors.ListedColormap(
    ['white', '#2ca02c', 'red', 'gray'])
boundaries = [0, 10, 20, 40, 50]
norm = mpl.colors.BoundaryNorm(boundaries, cmap.N, clip=True)

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
class3 = df.data['classifier'].copy()
class3[liquid] = 30

# maximum filter to increase the size of liquid region
liquid_max = maximum_filter(liquid, size=5)
# Median filter to remove background noise
liquid_smoothed = median_filter(liquid_max, size=13)

df.data['classifier'][liquid_smoothed] = 30
class4 = df.data['classifier'].copy()

# %%
first_plot = df.data['classifier'].copy()

# %%
# updraft - indication of aerosol zone
updraft = df.data['v_raw'] > 1
updraft_smooth = median_filter(updraft, size=3)
updraft_max = maximum_filter(updraft_smooth, size=91)

# Fill the gap in aerosol zone
updraft_median = median_filter(updraft_max, size=31)

# precipitation < -1 (center of precipitation)
precipitation_1 = (log_beta > -7) & (df.data['v_raw'] < -1)
class5 = df.data['classifier'].copy()
class5[precipitation_1] = 20
precipitation_1_median = median_filter(precipitation_1, size=9)

# Only select precipitation outside of aerosol zone
precipitation_1_ne = precipitation_1_median * ~updraft_median
precipitation_1_median_smooth = median_filter(precipitation_1_ne,
                                              size=3)
precipitation = precipitation_1_median_smooth
class6 = df.data['classifier'].copy()
class6[precipitation] = 20

# precipitation < -0.5 (include all precipitation)
precipitation_1_low = (log_beta > -7) & (df.data['v_raw'] < -0.5)

class7 = df.data['classifier'].copy()
class7[precipitation_1_low] = 20
# Avoid ebola infection surrounding updraft
# Useful to contain error during ebola precipitation
updraft_ebola = df.data['v_raw'] > 0.2
updraft_ebola_max = maximum_filter(updraft_ebola, size=3)

# %%
temp = df.data['classifier'].copy()

# Ebola precipitation
for i in range(1500):
    if i == 1:
        temp[precipitation] = 20
    prep_1_max = maximum_filter(precipitation, size=3)
    prep_1_max *= ~updraft_ebola_max  # Avoid updraft area
    precipitation_ = precipitation_1_low * prep_1_max
    if np.sum(precipitation) == np.sum(precipitation_):
        break
    precipitation = precipitation_

temp[precipitation] = 20
class8 = temp.copy()

# %%
second_plot = class8.copy()

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
third_plot = df.data['classifier'].copy()

# %%

class9 = df.data['classifier'].copy()


# %%
class10 = np.zeros(df.data['beta_raw'].shape, dtype=int)
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

    # df.data['classifier'][df.data['classifier'] == 10] = lab
    temp_ = db.labels_.copy()
    temp_[temp_ == 0] = np.max(db.labels_) + 1
    temp_[temp_ == -1] = np.max(db.labels_) + 2
    class10[df.data['classifier'] == 10] = temp_

# %%
df.data['classifier'][df.data['classifier'] == 10] = lab
class11 = df.data['classifier'].copy()
# %%
fourth_plot = df.data['classifier'].copy()


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
class12 = df.data['classifier'].copy()

# %%
fig, ax = plt.subplots(4, 1, figsize=(6.25, 6))
for ax_, data_plot in zip(ax.flatten(), [first_plot, second_plot, third_plot, fourth_plot]):
    p = ax_.pcolormesh(df.data['time'], df.data['range'],
                       data_plot.T,
                       cmap=cmap, norm=norm)
    ax_.yaxis.set_major_formatter(hd.m_km_ticks())
    ax_.set_ylabel('Height a.g.l [km]')
    ax_.set_yticks([0, 4000, 8000])
    cbar = fig.colorbar(p, ax=ax_, ticks=[5, 15, 30, 45])
    cbar.ax.set_yticklabels(['Background', 'Aerosol',
                             'Hydrometeor', 'Undefined'])
    ax_.set_xticks([0, 6, 12, 18, 24])
    ax_.set_xticklabels(['00:00', '06:00', '12:00', '18:00', '24:00'])
ax_.set_xlabel('Time UTC')
for n, ax_ in enumerate(ax.flatten()):
    ax_.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
             transform=ax_.transAxes, size=12)
fig.tight_layout()
fig.savefig(path + 'algorithm_demo.png', bbox_inches='tight')
