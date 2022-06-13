import xarray as xr
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
%matplotlib qt

# %%
data = hd.getdata('F:/halo/53/depolarization')
classifier_folder = 'F:\\halo\\classifier'
Path(classifier_folder).mkdir(parents=True, exist_ok=True)

# %%
date = '20190531'
file = [file for file in data if date in file][0]
df = hd.halo_data(file)

df.filter_height()
df.unmask999()
df.depo_cross_adj()

df.filter(variables=['beta_raw'],
          ref='co_signal',
          threshold=1 + df.snr_sd)

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

# # %%
# fig, axes = plt.subplots(6, 2, sharex=True, sharey=True,
#                          figsize=(16, 9))
# for val, ax, cmap_ in zip([aerosol, aerosol_smoothed,
#                            liquid_smoothed, precipitation_1_median,
#                            updraft_median,
#                            precipitation_1_median_smooth, precipitation_1_low,
#                            updraft_ebola_max, precipitation],
#                           axes.flatten()[2:-1],
#                           [['white', '#2ca02c'], ['white', '#2ca02c'],
#                            ['white', 'red'], ['white', 'blue'],
#                            ['white', '#D2691E'],
#                            ['white', 'blue'], ['white', 'blue'],
#                            ['white', '#D2691E'], ['white', 'blue']]):
#     ax.pcolormesh(df.data['time'], df.data['range'],
#                   val.T, cmap=mpl.colors.ListedColormap(cmap_))
# axes.flatten()[-1].pcolormesh(df.data['time'], df.data['range'],
#                               df.data['classifier'].T,
#                               cmap=cmap, norm=norm)
# axes[0, 0].pcolormesh(df.data['time'], df.data['range'],
#                       np.log10(df.data['beta_raw']).T,
#                       cmap='jet', vmin=-8, vmax=-4)
# axes[0, 1].pcolormesh(df.data['time'], df.data['range'],
#                       df.data['v_raw'].T, cmap='jet', vmin=-2, vmax=2)
# fig.tight_layout()
# fig.savefig(classifier_folder + '/' + df.filename + '_classifier.png',
#             dpi=150, bbox_inches='tight')

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
fig.savefig(classifier_folder + '/' + df.filename + '_classified.png',
            dpi=150, bbox_inches='tight')


# %% Despite the inefficient of loading data again, xarray is better at saving .nc files
dff = xr.open_dataset(file)

new_classifier = np.empty((df.data['time'].size, 3), dtype='int32')
new_classifier.fill(40)
dff['classified'] = (['time', 'range'],
                     np.concatenate((new_classifier, df.data['classifier']), axis=1))

new_depo = np.empty((df.data['time'].size, 3), dtype='float32')
new_depo.fill(-99)
dff['depo_bleed_corrected'] = (['time', 'range'],
                               np.concatenate((new_depo, df.data['depo_adj']), axis=1))

new_depo_sd = np.empty((df.data['time'].size, 3), dtype='float32')
new_depo_sd.fill(-99)
dff['depo_bleed_sd'] = (['time', 'range'],
                        np.concatenate((new_depo_sd, df.data['depo_adj_sd']), axis=1))

dff.attrs['classified'] = "Clasification algorithm by Vietle at github.com/vietle94/halo-lidar"
dff.attrs['bleed_corrected'] = "Bleed through corrected for depolarization ratio, see Vietle thesis"

dff.classified.attrs = {'units': ' ',
                        'long_name': 'Classified mask',
                        'comments': '0: Background, 10: Aerosol, 20: Precipitation, 30: Clouds, 40: Undefined'}
dff.depo_bleed_corrected.attrs = {'units': ' ',
                                  'long_name': 'Depolarization ratio (bleed through corrected)',
                                  'comments': 'Bleed through corrected'}
dff.depo_bleed_sd.attrs = {'units': ' ',
                           'long_name': 'Standard deviation of depolarization ratio (bleed through corrected)',
                           'comments': 'Bleed through corrected'}

dff.to_netcdf(classifier_folder + '/' + df.filename + '_classified.nc', format='NETCDF3_CLASSIC')
