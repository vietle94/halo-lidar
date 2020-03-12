# %% Load modules
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import halo_data as hd
from pathlib import Path
import csv
%matplotlib qt

# %%
data_folder = r'F:\halo\32\depolarization'
image_folder = r'F:\halo\32\depolarization\img'
depo_folder = r'F:\halo\32\depolarization\depo'
snr_folder = r'F:\halo\32\depolarization\snr'

for path in [image_folder, depo_folder, snr_folder]:
    Path(path).mkdir(parents=True, exist_ok=True)

# %%
# load whole folder data
data = hd.getdata(data_folder)

# %% get data
# pick day of data
file_name = next(data)
df = hd.halo_data(file_name)

# #
# # Some useful attributes and methods
# df.info
# df.full_data
# df.full_data_names
# #
# df.data
# # Names of data
# df.data_names
# # More info
# df.more_info
# # Get meta data of each variable
# df.meta_data('co_signal')
#
# # Get meta data of all variables
# {'==>' + key: df.meta_data(key) for key in df.full_data_names}
#
# # Only crucial info
# {'==>' + key: df.meta_data(key)['_attributes'] for key in df.full_data_names}

# %%
# Change masking missing values from -999 to NaN
df.unmask999()
# Remove first three columns of data matrix due to calibration,
# they correspond to top 3 height data
# ask Ville for more detail
# df.filter_height()
# Overview of data
df.describe()

# %%
# Plot data

image_raw = df.plot(
    variables=['beta_raw', 'v_raw', 'cross_signal', 'depo_raw', 'co_signal',
               'cross_signal_averaged', 'depo_averaged_raw', 'co_signal_averaged'],
    ncol=2, size=(20, 15))
image_raw.savefig(image_folder + '/' + df.filename + '_raw.png')

# %%
# Histogram of an area in SNR plot

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

# %%
# Calculate threshold
noise = area.area - 1
threshold = 1 + np.nanstd(noise) * 3
threshold

# %%
# Histogram of an area in SNR plot
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
# Append to or create new csv file
with open(snr_folder + '/' + df.filename + '.csv', 'a') as filedata:
    writer = csv.writer(filedata)
    for value in noise.flatten():
        writer.writerow([df.more_info['year'],
                         df.more_info['month'],
                         df.more_info['day'],
                         df.more_info['location'].decode('utf-8'),
                         df.more_info['systemID'],
                         value])

with open(snr_folder + '/' + df.filename + '_avg' + '.csv', 'a') as filedata:
    writer = csv.writer(filedata)
    for value in noise_averaged.flatten():
        writer.writerow([df.more_info['year'],
                         df.more_info['month'],
                         df.more_info['day'],
                         df.more_info['location'].decode('utf-8'),
                         df.more_info['systemID'],
                         value])

# %%
df.filter(variables=['beta_raw', 'v_raw', 'cross_signal', 'depo_raw'],
          ref='co_signal', threshold=threshold)

df.filter(variables=['cross_signal_averaged', 'depo_averaged_raw'],
          ref='co_signal_averaged', threshold=threshold_averaged)

# %%
# Plot data
image_filtered = df.plot(
    variables=['beta_raw', 'v_raw', 'cross_signal', 'depo_raw', 'co_signal',
               'cross_signal_averaged', 'depo_averaged_raw', 'co_signal_averaged'],
    ncol=2, size=(20, 15))
image_filtered.savefig(image_folder + '/' + df.filename + '_filtered.png')

# %% Summary
df.describe()

# %%
# Area selection for depo cloud
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
ax1.set_title('depo')
fig.suptitle(df.filename,
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

# %%
# Extract data from each time point
i = area.i
area_value = area.area[:, i]
area_range = df.data['range'][area.maskrange]
area_snr = df.data['co_signal'].transpose()[area.mask][:, i]
area_vraw = df.data['v_raw'].transpose()[area.mask][:, i]
area_betaraw = df.data['beta_raw'].transpose()[area.mask][:, i]
area_cross = df.data['cross_signal'].transpose()[area.mask][:, i]

# Calculate indice of maximum snr value
max_i = np.argmax(area_snr)

result = [df.more_info['year'],
          df.more_info['month'],
          df.more_info['day'],
          df.more_info['location'].decode('utf-8'),
          df.more_info['systemID'],
          df.data['time'][area.masktime][i],  # time as hour
          area_range[max_i],  # range
          area_value[max_i],  # depo value
          area_snr[max_i],  # snr
          area_vraw[max_i],  # v_raw
          area_betaraw[max_i],  # beta_raw
          area_cross[max_i]  # cross_signal
          ]
# Append to or create new csv file
with open(depo_folder + '/' + df.filename + '.csv', 'a') as filedata:
    writer = csv.writer(filedata)
    writer.writerow(result)
