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
data = hd.getdata('F:/halo/46/depolarization')
snr = glob.glob('F:/halo/46/depolarization/snr/*_noise.csv')
classifier_folder = 'F:\\halo\\classifier'
Path(classifier_folder).mkdir(parents=True, exist_ok=True)

# %%
date = '20180514'
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
for _ in range(500):
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


# fig, axes = plt.subplots(6, 2, sharex=True, sharey=True,
#                          figsize=(16, 9))
# for val, ax, cmap in zip([aerosol, aerosol_smoothed,
#                           liquid_smoothed, precipitation_1_median,
#                           updraft_median,
#                           precipitation_1_median_smooth, precipitation_1_low,
#                           updraft_ebola_max, precipitation],
#                          axes.flatten()[2:-1],
#                          [['white', '#2ca02c'], ['white', '#2ca02c'],
#                           ['white', 'red'], ['white', 'blue'],
#                           ['white', '#D2691E'],
#                           ['white', 'blue'], ['white', 'blue'],
#                           ['white', '#D2691E'], ['white', 'blue']]):
#     ax.pcolormesh(df.data['time'], df.data['range'],
#                   val.T, cmap=mpl.colors.ListedColormap(cmap))
# axes.flatten()[-1].pcolormesh(df.data['time'], df.data['range'],
#                               df.data['classifier'].T,
#                               cmap=mpl.colors.ListedColormap(
#     ['white', '#2ca02c', 'blue', 'red']),
#     vmin=0, vmax=3)
# axes[0, 0].pcolormesh(df.data['time'], df.data['range'],
#                       np.log10(df.data['beta_raw']).T,
#                       cmap='jet', vmin=-8, vmax=-4)
# axes[0, 1].pcolormesh(df.data['time'], df.data['range'],
#                       df.data['v_raw'].T, cmap='jet', vmin=-2, vmax=2)
# fig.tight_layout()
# fig.savefig(classifier_folder + '/' + df.filename + '_classifier.png',
#             dpi=150, bbox_inches='tight')

# %%
classifier = df.data['classifier'].flatten()
time_save = np.repeat(df.data['time'],
                      df.data['beta_raw'].shape[1])
range_save = np.tile(df.data['range'],
                     df.data['beta_raw'].shape[0])
v_save = df.data['v_raw'].flatten()  # put here to avoid noisy values at 1sd snr
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
#
# result.to_csv(classifier_folder + '/' + df.filename + '_classified.csv',
#               index=False)

# %%
temp = df.data['classifier']
filter_c = df.data['classifier'] != 0
df = hd.halo_data(file)
df.filter_height()
df.unmask999()
df.depo_cross_adj()
df.moving_average(n=51)
df.beta_averaged()

filter = temp != 10
co = df.data['co_signal'].copy()
co[filter] = np.nan
co = hd.ma(co, n=101)
cross = df.data['cross_signal'].copy()
cross[filter] = np.nan
cross = hd.ma(cross, n=101)
depo = (cross - 1) / (co - 1)
for val in ['beta_averaged', 'v_raw', 'depo_adj_averaged']:
    df.data[val] = np.where(filter_c, df.data[val], np.nan)
df.data['classifier'] = temp

# %%


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


cmap = plt.get_cmap('Greens')
new_cmap = truncate_colormap(cmap, 0.15, 1)

dep = depo.copy()
dep[df.data['classifier'] != 10] = np.nan
fig, ax = plt.subplots(figsize=(12, 3))
ax.pcolormesh(df.data['time'], df.data['range'],
              df.data['classifier'].T, cmap=mpl.colors.ListedColormap(
    ['white', 'white', 'blue', 'red']))
p = ax.pcolormesh(df.data['time'], df.data['range'],
                  dep.T, cmap='jet', vmin=0, vmax=0.5)
fig.colorbar(p, ax=ax)

# %%
fig, ax = plt.subplots()
depo_hist = dep[(dep < 0.4) & (dep > -0.25)]
ax.hist(depo_hist.flatten(), bins=40)

# %%
fig = plt.figure(figsize=(16, 9))
ax1 = fig.add_subplot(421)
ax2 = fig.add_subplot(422)
ax3 = fig.add_subplot(423, sharex=ax1, sharey=ax1)
ax4 = fig.add_subplot(424, sharex=ax2)
ax5 = fig.add_subplot(425, sharex=ax1, sharey=ax1)
ax6 = fig.add_subplot(426, sharex=ax2)
ax7 = fig.add_subplot(427, sharex=ax1, sharey=ax1)
ax8 = fig.add_subplot(428, sharex=ax2)
ax1.pcolormesh(df.data['time'], df.data['range'],
               np.log10(df.data['beta_averaged']).T, cmap='jet', vmin=-8, vmax=-4)
ax3.pcolormesh(df.data['time'], df.data['range'],
               df.data['v_raw'].T, cmap='jet', vmin=-2, vmax=2)
ax5.pcolormesh(df.data['time'], df.data['range'],
               df.data['depo_adj_averaged'].T, cmap='jet', vmin=0, vmax=0.5)
ax7.pcolormesh(df.data['time'], df.data['range'],
               df.data['classifier'].T,
               cmap=mpl.colors.ListedColormap(
    ['white', '#2ca02c', 'blue', 'red']))

bin_time = np.arange(0, 24+0.35, 0.25)
bin_height = np.arange(0, df.data['range'].max() + 31, 30)
for i, ax, lab in zip([10, 20, 30], [ax2, ax4, ax6],
                      ['aerosol_15min', 'precipitation', 'clouds']):
    ax.set_ylabel(lab)
    if (classifier == i).any():
        co, _, _, _ = binned_statistic_2d(range_save[classifier == i],
                                          time_save[classifier == i],
                                          co_save[classifier == i],
                                          bins=[bin_height, bin_time],
                                          statistic=np.nanmean)
        cross, _, _, _ = binned_statistic_2d(range_save[classifier == i],
                                             time_save[classifier == i],
                                             cross_save[classifier == i],
                                             bins=[bin_height, bin_time],
                                             statistic=np.nanmean)
        depo = (cross-1)/(co-1)
        depo = depo[depo < 0.8]
        depo = depo[depo > -0.25]
        ax.hist(depo, bins=40)

ax8.set_ylabel('aerosol_1hr')
if (classifier == 10).any():
    bin_time1h = np.arange(0, 24+1.5, 1)
    co, _, _, _ = binned_statistic_2d(range_save[classifier == 10],
                                      time_save[classifier == 10],
                                      co_save[classifier == 10],
                                      bins=[bin_height, bin_time1h],
                                      statistic=np.nanmean)
    cross, _, _, _ = binned_statistic_2d(range_save[classifier == 10],
                                         time_save[classifier == 10],
                                         cross_save[classifier == 10],
                                         bins=[bin_height, bin_time1h],
                                         statistic=np.nanmean)
    depo = (cross-1)/(co-1)
    depo = depo[depo < 0.8]
    depo = depo[depo > -0.25]
    ax8.hist(depo, bins=40)


fig.tight_layout()
# fig.savefig(classifier_folder + '/' + df.filename + '_hist.png',
#             dpi=150, bbox_inches='tight')

# %%
time_c = np.repeat(np.arange(df.data['time'].size),
                   df.data['beta_raw'].shape[1])
height_c = np.tile(np.arange(df.data['range'].size),
                   df.data['beta_raw'].shape[0])

time_c = time_c[classifier == 10].reshape(-1, 1)
height_c = height_c[classifier == 10].reshape(-1, 1)
X = np.hstack([time_c, height_c])
db = DBSCAN(eps=3, min_samples=1).fit(X)

v_c = v_save[classifier == 10]
range_c = range_save[classifier == 10]

# %%
temp = df.data['classifier'].copy()
v_dict = {}
r_dict = {}
for i in np.unique(db.labels_):
    v_dict[i] = np.nanmean(v_c[db.labels_ == i])
    r_dict[i] = np.nanmin(range_c[db.labels_ == i])

# %%
lab = db.labels_.copy()
for key, val in v_dict.items():
    if (val < -0.5):
        lab[db.labels_ == key] = 20
    elif r_dict[key] == min(df.data['range']):
        lab[db.labels_ == key] = 10
    elif (val > -0.2):
        lab[db.labels_ == key] = 11
    else:
        lab[db.labels_ == key] = 31

temp[df.data['classifier'] == 10] = lab

# %%
ground = temp == 20
ground[:, 3:] = False
rain = temp == 20
for _ in range(500):
    ground_max = maximum_filter(ground, size=3)
    ground_rain = ground_max * rain
    if np.sum(ground_rain) == np.sum(ground):
        break
    ground = ground_rain
temp[ground_rain] = 21

# %%
cmap = mpl.colors.ListedColormap(
    ['red', '#000000', '#ffffff', 'blue', 'orange', 'green', 'yellow'])
boundaries = [0, 10, 11, 20, 21, 30, 31, 40]
norm = mpl.colors.BoundaryNorm(boundaries, cmap.N, clip=True)
fig, ax = plt.subplots(figsize=(12, 3))
p = ax.pcolormesh(df.data['time'], df.data['range'],
                  temp.T, cmap=cmap, norm=norm)
fig.colorbar(p, ax=ax)
# ax.pcolormesh(df.data['time'], df.data['range'],
#               dep.T, cmap='summer', vmin=0, vmax=0.5)
