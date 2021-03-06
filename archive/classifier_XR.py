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
data = hd.getdata('F:/halo/32/depolarization')
classifier_folder = 'F:\\halo\\classifier'
Path(classifier_folder).mkdir(parents=True, exist_ok=True)
with open('ref_XR2.npy', 'rb') as f:
    ref = np.load(f)

# %%
date = '20180908'
file = [file for file in data if date in file][0]
df = hd.halo_data(file)

df.filter_height()
df.unmask999()
df.depo_cross_adj()

# %% if XR
df.data['beta_log'] = np.log10(df.data['beta_raw'])
df.data['beta_log'][:, :50] = df.data['beta_log'][:, :50] - ref

# %%
df.filter(variables=['beta_raw', 'beta_log'],
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
                           beta=df.data['beta_log'],
                           v=df.data['v_raw'])

# Small size median filter to remove noise
aerosol_smoothed = median_filter(aerosol, size=11)
df.data['classifier'][aerosol_smoothed] = 10

df.filter(variables=['beta_raw', 'v_raw', 'depo_adj', 'beta_log'],
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
                          beta=df.data['beta_log'],
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
                           beta=df.data['beta_log'],
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
                                   beta=df.data['beta_log'],
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
                                       beta=df.data['beta_log'],
                                       v=df.data['v_raw'])

# Avoid ebola infection surrounding updraft
# Useful to contain error during ebola precipitation
updraft_ebola = df.decision_tree(depo_thres=[None, None],
                                 beta_thres=[None, None],
                                 v_thres=[0.2, None],
                                 depo=df.data['depo_adj'],
                                 beta=df.data['beta_log'],
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
if (df.data['classifier'] == 10).any():
    classifier = df.data['classifier'].ravel()
    time_dbscan = np.repeat(np.arange(df.data['time'].size),
                            df.data['beta_raw'].shape[1])
    height_dbscan = np.tile(np.arange(df.data['range'].size),
                            df.data['beta_raw'].shape[0])

    time_dbscan = time_dbscan[classifier == 10].reshape(-1, 1)
    height_dbscan = height_dbscan[classifier == 10].reshape(-1, 1)
    X = np.hstack([time_dbscan, height_dbscan])
    db = DBSCAN(eps=3, min_samples=1, n_jobs=-1).fit(X)

    v_dbscan = v_save[classifier == 10]
    range_dbscan = range_save[classifier == 10]

    v_dict = {}
    r_dict = {}
    for i in np.unique(db.labels_):
        v_dict[i] = np.nanmean(v_dbscan[db.labels_ == i])
        r_dict[i] = np.nanmin(range_dbscan[db.labels_ == i])

    lab = db.labels_.copy()
    for key, val in v_dict.items():
        if (val < -0.5):
            lab[db.labels_ == key] = 20
        elif r_dict[key] == min(df.data['range']):
            lab[db.labels_ == key] = 10
        elif (val > -0.2):
            lab[db.labels_ == key] = 11
        else:
            lab[db.labels_ == key] = 40

    df.data['classifier'][df.data['classifier'] == 10] = lab

# %%
ground = df.data['classifier'] == 20
ground[:, 3:] = False
rain = df.data['classifier'] == 20
for _ in range(1500):
    ground_max = maximum_filter(ground, size=3)
    ground_rain = ground_max * rain
    if np.sum(ground_rain) == np.sum(ground):
        break
    ground = ground_rain
df.data['classifier'][ground] = 21

# %%
cmap = mpl.colors.ListedColormap(
    ['white', '#2ca02c', '#808000', '#00ffff', 'blue', 'red', 'gray'])
boundaries = [0, 10, 11, 20, 21, 30, 40, 50]
norm = mpl.colors.BoundaryNorm(boundaries, cmap.N, clip=True)

# %%
fig, axes = plt.subplots(6, 2, sharex=True, sharey=True,
                         figsize=(16, 9))
for val, ax, cmap_ in zip([aerosol, aerosol_smoothed,
                           liquid_smoothed, precipitation_1_median,
                           updraft_median,
                           precipitation_1_median_smooth, precipitation_1_low,
                           updraft_ebola_max, precipitation],
                          axes.flatten()[2:-1],
                          [['white', '#2ca02c'], ['white', '#2ca02c'],
                           ['white', 'red'], ['white', 'blue'],
                           ['white', '#D2691E'],
                           ['white', 'blue'], ['white', 'blue'],
                           ['white', '#D2691E'], ['white', 'blue']]):
    ax.pcolormesh(df.data['time'], df.data['range'],
                  val.T, cmap=mpl.colors.ListedColormap(cmap_))
axes.flatten()[-1].pcolormesh(df.data['time'], df.data['range'],
                              df.data['classifier'].T,
                              cmap=cmap, norm=norm)
axes[0, 0].pcolormesh(df.data['time'], df.data['range'],
                      np.log10(df.data['beta_raw']).T,
                      cmap='jet', vmin=-8, vmax=-4)
axes[0, 1].pcolormesh(df.data['time'], df.data['range'],
                      df.data['v_raw'].T, cmap='jet', vmin=-2, vmax=2)
fig.tight_layout()
fig.savefig(classifier_folder + '/' + df.filename + '_classifier.png',
            dpi=150, bbox_inches='tight')

# %%
classifier = df.data['classifier'].flatten()
result = pd.DataFrame({'date': df.date,
                       'location': df.location,
                       'beta': np.log10(beta_save),
                       'v': v_save,
                       'depo': depo_save,
                       'co_signal': co_save,
                       'cross_signal': cross_save,
                       'time': time_save,
                       'range': range_save,
                       'classifier': classifier})

result.to_csv(classifier_folder + '/' + df.filename + '_classified.csv',
              index=False)

# %%
fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(4, 2, width_ratios=[5, 4])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0], sharex=ax1, sharey=ax1)
ax4 = fig.add_subplot(gs[1, 1], sharex=ax2)
ax5 = fig.add_subplot(gs[2, 0], sharex=ax1, sharey=ax1)
ax6 = fig.add_subplot(gs[2, 1], sharex=ax2)
ax7 = fig.add_subplot(gs[3, 0], sharex=ax1, sharey=ax1)
ax8 = fig.add_subplot(gs[3, 1], sharex=ax2)
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
    ax.set_ylabel('Height [km, a.g.l]')

cbar = fig.colorbar(p1, ax=ax1)
cbar.ax.set_ylabel('Attenuated backscatter')
cbar.ax.yaxis.set_label_position('left')
cbar = fig.colorbar(p2, ax=ax3)
cbar.ax.set_ylabel('Velocity [m/s]')
cbar.ax.yaxis.set_label_position('left')
cbar = fig.colorbar(p3, ax=ax5)
cbar.ax.set_ylabel('Depolarization ratio')
cbar.ax.yaxis.set_label_position('left')
cbar = fig.colorbar(p4, ax=ax7, ticks=[5, 10.5, 15, 20.5, 25, 35, 45])
cbar.ax.set_yticklabels(['Background', 'Aerosol', 'Elevated aerosol', 'Ice clouds',
                         'Precipitation', 'Clouds', 'Undefined'])

for i, ax, lab in zip([20, 21, 30], [ax2, ax4, ax6],
                      ['ice clouds', 'precipitation', 'clouds']):
    ax.set_ylabel(lab)
    if (classifier == i).any():
        depo = depo_save[classifier == i]
        depo = depo[depo < 0.8]
        depo = depo[depo > -0.25]
        ax.hist(depo, bins=40)

bin_time = np.arange(0, 24+0.35, 0.25)
bin_height = np.arange(0, df.data['range'].max() + 31, 30)

ax8.set_ylabel('aerosol_30min')
if (classifier // 10 == 1).any():
    bin_time1h = np.arange(0, 24+0.5, 0.5)
    co, _, _, _ = binned_statistic_2d(range_save[classifier // 10 == 1],
                                      time_save[classifier // 10 == 1],
                                      co_save[classifier // 10 == 1],
                                      bins=[bin_height, bin_time1h],
                                      statistic=np.nanmean)
    cross, _, _, _ = binned_statistic_2d(range_save[classifier // 10 == 1],
                                         time_save[classifier // 10 == 1],
                                         cross_save[classifier // 10 == 1],
                                         bins=[bin_height, bin_time1h],
                                         statistic=np.nanmean)
    depo = (cross-1)/(co-1)
    depo = depo[depo < 0.8]
    depo = depo[depo > -0.25]
    ax8.hist(depo, bins=40)

ax8.set_xlabel('Depolarization ratio', weight='bold')
ax7.set_xlabel('Time UTC [hour]', weight='bold')

fig.tight_layout()
fig.savefig(classifier_folder + '/' + df.filename + '_hist.png',
            dpi=150, bbox_inches='tight')

# %%
fig, ax = plt.subplots(2, 1, figsize=(12, 6))
ax[0].pcolormesh(df.data['time'], df.data['range'],
                 np.log10(df.data['beta_raw']).T, cmap='jet', vmin=-8, vmax=-4)
ax[1].pcolormesh(df.data['time'], df.data['range'],
                 df.data['beta_log'].T, cmap='jet', vmin=-8, vmax=-4)
