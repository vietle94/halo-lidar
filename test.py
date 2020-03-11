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
# Remove first three columns of data matrix due to calibration,
# they correspond to top 3 height data
# ask Ville for more detail
df.filter_height()
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
area = hd.area_select(df.data['time'],
                      df.data['range'],
                      df.data['co_signal'].transpose(),
                      ax[0],
                      ax[1],
                      type='kde')
fig.colorbar(p, ax=ax[0])
# %% Calculate threshold
noise = area.area - 1
threshold = 1 + np.nanstd(noise) * 3

np.nanstd(noise)
threshold

# %% Histogram of an area in SNR plot
%matplotlib qt
fig, ax = plt.subplots(1, 2, figsize=(24, 12))
p = ax[0].pcolormesh(df.data['time_averaged'],
                     df.data['range'],
                     df.data['co_signal_averaged'].transpose(),
                     cmap='jet', vmin=0.995, vmax=1.005)
area = hd.area_select(df.data['time_averaged'],
                      df.data['range'],
                      df.data['co_signal_averaged'].transpose(),
                      ax[0],
                      ax[1],
                      type='kde')
fig.colorbar(p, ax=ax[0])
# %% Calculate threshold
noise_averaged = area.area - 1
threshold_averaged = 1 + np.nanstd(noise_averaged) * 2

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
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(24, 12))
ax = ax.flatten()
for i, name in enumerate(['depo_raw', 'beta_raw', 'v_raw']):
    p = ax[i].pcolormesh(df.data['time'],
                         df.data['range'],
                         df.data[name].transpose() if name != 'beta_raw' else np.log10(
                             df.data[name].transpose()),
                         cmap='jet', vmin=df.cbar_lim[name][0], vmax=df.cbar_lim[name][1])
    ax[i].set_title(name)
    fig.colorbar(p, ax=ax[i])
area = hd.area_select(df.data['time'],
                      df.data['range'],
                      df.data['depo_raw'].transpose(),
                      ax[0],
                      type=None)
plt.tight_layout()
# %%
i = -1
final_result = pd.DataFrame(columns=['time', 'range', 'SNR', 'depo'])
# %%
i += 1
%matplotlib inline
fig, ax = plt.subplots(figsize=(8, 5))
area_value = area.area[:, i]
area_range = df.data['range'][area.maskrange]
area_snr = df.data['co_signal'].transpose()[area.mask][:, i]
p = ax.scatter(area_value, area_range,
               c=area_snr,
               s=area_snr * 20)
ax.set_title(f"Depo value colored by SNR at time {df.data['time'][area.masktime][i]:.3f}")
ax.set_xlabel('Depo value')
ax.set_ylabel('Range')
fig.colorbar(p)


# %%
max_i = np.argmax(area_snr)
result = pd.Series({'time': df.data['time'][area.masktime][i],
                    'range': area_range[max_i],
                    'SNR': area_snr[max_i],
                    'depo': area_value[max_i]})
result
final_result = final_result.append(result, ignore_index=True)
final_result

# %%
plt.plot(final_result['depo'], '-')

# %%
# Location etc
name = [int(df.more_info.get(key)) if key != 'location' else
        df.more_info.get(key).decode("utf-8") for
        key in ['location', 'year', 'month', 'day', 'systemID']]

filename = '-'.join([str(elem) for elem in name])

final_result.to_csv(filename + '.csv')
