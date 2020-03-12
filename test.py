# %% Load modules
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import halo_data as hd
%matplotlib qt
# %%
# load whole folder data
data = hd.getdata("C:/Users/LV/OneDrive - University of Helsinki/FMI/halo/53/depolarization/")
# data = hd.getdata("G:/OneDrive - University of Helsinki/FMI/halo/53/depolarization/")
# data = hd.getdata(r'G:\OneDrive - University of Helsinki\FMI\halo\53\depolarization')

# %% get data
# pick day of data
file_name = next(data)
df = hd.halo_data(file_name)
df.info
df.full_data
df.full_data_names
#
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
# df.unmask999()
# Remove first three columns of data matrix due to calibration,
# they correspond to top 3 height data
# ask Ville for more detail
df.filter_height()
# Overview of data
df.describe()

# %%
# Plot data

df.plot(
    variables=['beta_raw', 'v_raw', 'cross_signal', 'depo_raw', 'co_signal',
               'cross_signal_averaged', 'depo_averaged_raw', 'co_signal_averaged'], ncol=2, size=(20, 15))

# %% Histogram of an area in SNR plot

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

df.plot(
    variables=['beta_raw', 'v_raw', 'cross_signal', 'depo_raw', 'co_signal',
               'cross_signal_averaged', 'depo_averaged_raw', 'co_signal_averaged'], ncol=2, size=(20, 15))


# %% Summary
df.describe()

# %%

final_result = pd.DataFrame(columns=['time', 'range', 'SNR', 'depo'])
# %%

fig = plt.figure(figsize=(24, 12))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(223)
ax3 = fig.add_subplot(224, sharey=ax2)
p = ax1.pcolormesh(df.data['time'],
                   df.data['range'],
                   np.log10(df.data['beta_raw'].transpose()),
                   cmap='jet', vmin=df.cbar_lim['beta_raw'][0],
                   vmax=df.cbar_lim['beta_raw'][1])
me = fig.colorbar(p, ax=ax1)
ax1.set_title(df.full_data.filename.split('\\')[-1].split('_')[0] + ' - ' +
              df.more_info['location'].decode("utf-8") + ' - ' +
              str(df.more_info['systemID']),
              size=30,
              weight='bold')

area = hd.area_select(df.data['time'],
                      df.data['range'],
                      df.data['depo_raw'].transpose(),
                      ax1,
                      ax2,
                      ax3,
                      type='time_point',
                      ref=df.data['co_signal'])

final_result = pd.DataFrame(columns=['time', 'range', 'SNR', 'depo'])

# %%
i = area.i
area_value = area.area[:, i]
area_range = df.data['range'][area.maskrange]
area_snr = df.data['co_signal'].transpose()[area.mask][:, i]

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
