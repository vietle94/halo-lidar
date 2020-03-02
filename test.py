# %% Load modules
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import halo_data as hd

# %%
data = hd.getdata("C:/Users/LV/OneDrive - University of Helsinki/FMI/halo/53/depolarization/")
# data = hd.getdata("G:/OneDrive - University of Helsinki/FMI/halo/53/depolarization/")
# data = hd.getdata(r'G:\OneDrive - University of Helsinki\FMI\halo\53\depolarization')

# %% get data
file_name = next(data)
df = hd.halo_data(file_name)
df.info
df.full_data
df.full_data_names
df.data
# Names of data
df.data_names
# More info
df.more_info
# Get meta data of each variable
df.meta_data('co_signal')

# Get meta data of all variables
{'==>' + key: df.meta_data(key) for key in df.full_data_names}

# Only crucial info
{'==>' + key: df.meta_data(key)['_attributes'] for key in df.full_data_names}

# %%
# Change masking missing values from -999 to NaN
df.unmask999()
# Overview of data
df.describe()

# %%
# Plot data
%matplotlib inline
df.plot(
    variables=['beta_raw', 'v_raw', 'cross_signal', 'depo_raw', 'co_signal',
               'cross_signal_averaged', 'depo_averaged_raw', 'co_signal_averaged'], ncol=2, size=(20, 15))

# %% Histogram of an area in SNR plot
%matplotlib qt
fig, ax = plt.subplots(1, 2, figsize=(24, 12))
p = ax[0].pcolormesh(df.data['time'],
                     df.data['range'],
                     df.data['co_signal'].transpose(),
                     cmap='jet', vmin=0.995, vmax=1.005)
area = hd.area_histogram(ax[0], ax[1], fig, df.data['time'],
                         df.data['range'],
                         df.data['co_signal'].transpose(),
                         hist=False)
fig.colorbar(p, ax=ax[0])
# %% Calculate threshold
noise = area.area - 1
threshold = 1 + np.nanmean(noise) + np.nanstd(noise) * 2

threshold

# %% Histogram of an area in SNR plot
%matplotlib qt
fig, ax = plt.subplots(1, 2, figsize=(24, 12))
p = ax[0].pcolormesh(df.data['time_averaged'],
                     df.data['range'],
                     df.data['co_signal_averaged'].transpose(),
                     cmap='jet', vmin=0.995, vmax=1.005)
area = hd.area_histogram(ax[0], ax[1], fig, df.data['time_averaged'],
                         df.data['range'],
                         df.data['co_signal_averaged'].transpose(),
                         hist=False)
fig.colorbar(p, ax=ax[0])
# %% Calculate threshold
noise_averaged = area.area - 1
threshold_averaged = 1 + np.nanmean(noise_averaged) + np.nanstd(noise_averaged) * 2

threshold_averaged


# %%
df.filter(variables=['beta_raw', 'v_raw', 'cross_signal', 'depo_raw'],
          ref='co_signal', threshold=threshold)

df.filter(variables=['cross_signal_averaged', 'depo_averaged_raw'],
          ref='co_signal_averaged', threshold=threshold_averaged)

# %%
# Plot data
%matplotlib inline
df.plot(
    variables=['beta_raw', 'v_raw', 'cross_signal', 'depo_raw', 'co_signal',
               'cross_signal_averaged', 'depo_averaged_raw', 'co_signal_averaged'], ncol=2, size=(20, 15))


# %% Summary
df.describe()

# %%
%matplotlib qt
fig, ax = plt.subplots(2, 2, figsize=(24, 12))
ax = ax.flatten()
for i, name in zip([0, 2, 3], ['depo_raw', 'beta_raw', 'co_signal']):
    p = ax[i].pcolormesh(df.data['time'],
                         df.data['range'],
                         df.data[name].transpose() if name != 'beta_raw' else np.log10(
                             df.data[name].transpose()),
                         cmap='jet', vmin=df.cbar_lim[name][0], vmax=df.cbar_lim[name][1])
    ax[i].set_title(name)
    fig.colorbar(p, ax=ax[i])
area = hd.area_histogram(ax[0], ax[1], fig, df.data['time'],
                         df.data['range'],
                         df.data['depo_raw'].transpose(),
                         hist=False)
