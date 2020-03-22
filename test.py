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
# Specify data folder path
data_folder = r'F:\halo\32\depolarization'
# Specify output images folder path
image_folder = r'F:\halo\32\depolarization\img'
# Specify folder path for output depo analysis
depo_folder = r'F:\halo\32\depolarization\depo'
# Specify folder path for snr collection
snr_folder = r'F:\halo\32\depolarization\snr'
# Make those specified folders if they are not existed yet
for path in [image_folder, depo_folder, snr_folder]:
    Path(path).mkdir(parents=True, exist_ok=True)

# %%
# Get a list of all files in data folder
data = hd.getdata(data_folder)

# %%
# pick date of data
pick_date = '20170609'
data_indices = hd.getdata_date(data, pick_date)
print(data[data_indices])
data_indices = data_indices - 1

# %%
##################################################
#
# START HERE to go to next data, if you start a new session,
# start from the beginning and remember to to pick date of data
#
##################################################
# Load data
plt.close(fig='all')
data_indices = data_indices + 1
df = hd.halo_data(data[data_indices])

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
%matplotlib qt
image_raw = df.plot(
    variables=['beta_raw', 'v_raw', 'cross_signal', 'depo_raw', 'co_signal',
               'cross_signal_averaged', 'depo_averaged_raw', 'co_signal_averaged'],
    ncol=2, size=(18, 9))
image_raw.savefig(image_folder + '/' + df.filename + '_raw.png')

# %%
# Close the plot
plt.close(fig=image_raw)

# %%
# Histogram of an area in SNR plot
fig, ax = plt.subplots(1, 2, figsize=(18, 9))
p = ax[0].pcolormesh(df.data['time'],
                     df.data['range'],
                     df.data['co_signal'].transpose(),
                     cmap='jet', vmin=0.995, vmax=1.005)
ax[0].yaxis.set_major_formatter(hd.m_km_ticks())
ax[0].set_title('Choose background noise for SNR')
ax[0].set_ylabel('Height (km)')
ax[0].set_xlabel('Time (h)')
ax[0].set_ylim(bottom=0)
area = hd.area_snr(df.data['time'],
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
# Close the plot
plt.close(fig=fig)

# %%
# Histogram of an area in SNR plot
fig, ax = plt.subplots(1, 2, figsize=(18, 9))
p = ax[0].pcolormesh(df.data['time_averaged'],
                     df.data['range'],
                     df.data['co_signal_averaged'].transpose(),
                     cmap='jet', vmin=0.995, vmax=1.005)
ax[0].yaxis.set_major_formatter(hd.m_km_ticks())
ax[0].set_title('Choose background noise for SNR averaged')
ax[0].set_ylabel('Height (km)')
ax[0].set_xlabel('Time (h)')
ax[0].set_ylim(bottom=0)
area = hd.area_snr(df.data['time_averaged'],
                   df.data['range'],
                   df.data['co_signal_averaged'].transpose(),
                   ax[0],
                   ax[1],
                   type='kde')
fig.colorbar(p, ax=ax[0])
# %% Calculate threshold
noise_averaged = area.area - 1
threshold_averaged = 1 + np.nanstd(noise_averaged) * 3
threshold_averaged

# %%
# Close the plot
plt.close(fig=fig)

# %%
# Append to or create new csv file
with open(snr_folder + '/' + df.filename + '_noise.csv', 'w') as f:
    noise_shape = noise.flatten().shape
    noise_csv = pd.DataFrame.from_dict({'year': np.repeat(df.more_info['year'], noise_shape),
                                        'month': np.repeat(df.more_info['month'], noise_shape),
                                        'day': np.repeat(df.more_info['day'], noise_shape),
                                        'location': np.repeat(df.more_info['location'].decode('utf-8'), noise_shape),
                                        'systemID': np.repeat(df.more_info['systemID'], noise_shape),
                                        'noise': noise.flatten()})
    noise_csv.to_csv(f, header=f.tell() == 0, index=False)

with open(snr_folder + '/' + df.filename + '_noise_avg' + '.csv', 'w') as f:
    noise_avg_shape = noise_averaged.flatten().shape
    noise_avg_csv = pd.DataFrame.from_dict({'year': np.repeat(df.more_info['year'], noise_avg_shape),
                                            'month': np.repeat(df.more_info['month'], noise_avg_shape),
                                            'day': np.repeat(df.more_info['day'], noise_avg_shape),
                                            'location': np.repeat(df.more_info['location'].decode('utf-8'), noise_avg_shape),
                                            'systemID': np.repeat(df.more_info['systemID'], noise_avg_shape),
                                            'noise': noise_averaged.flatten()})
    noise_avg_csv.to_csv(f, header=f.tell() == 0, index=False)
# Remove unuse variables
del noise_csv, noise_shape, noise_avg_csv, noise_avg_shape

# %%
# Use those obtained threshold to filter data, this will overwrite raw data
df.filter(variables=['beta_raw', 'v_raw', 'cross_signal', 'depo_raw'],
          ref='co_signal', threshold=threshold)

df.filter(variables=['cross_signal_averaged', 'depo_averaged_raw'],
          ref='co_signal_averaged', threshold=threshold_averaged)

# %%
# Plot filtered data
image_filtered = df.plot(
    variables=['beta_raw', 'v_raw', 'cross_signal', 'depo_raw', 'co_signal',
               'cross_signal_averaged', 'depo_averaged_raw', 'co_signal_averaged'],
    ncol=2, size=(18, 9))
image_filtered.savefig(image_folder + '/' + df.filename + '_filtered.png')

# %%
##################################################
#
# STOP HERE if you don't like any clouds
#
##################################################
# Close the plot
plt.close(fig=image_filtered)

# %% Summary
df.describe()

# %%
# Area selection for depo cloud
fig = plt.figure(figsize=(18, 9))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(223)
ax3 = fig.add_subplot(224, sharey=ax2)
p = ax1.pcolormesh(df.data['time'],
                   df.data['range'],
                   np.log10(df.data['beta_raw'].transpose()),
                   cmap='jet', vmin=df.cbar_lim['beta_raw'][0],
                   vmax=df.cbar_lim['beta_raw'][1])
me = fig.colorbar(p, ax=ax1, fraction=0.05, pad=0.02)
ax1.set_title('beta_raw')
ax1.set_xlabel('Time (h)')
ax1.set_xlim([0, 24])
ax1.set_ylabel('Height (km)')
ax1.yaxis.set_major_formatter(hd.m_km_ticks())
fig.suptitle(df.filename,
             size=30,
             weight='bold')
area = hd.area_timeprofile(df.data['time'],
                           df.data['range'],
                           df.data['depo_raw'].transpose(),
                           ax1,
                           ax_snr=ax3,
                           ax_depo=ax2,
                           snr=df.data['co_signal'])

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

result = pd.DataFrame.from_dict([{
    'year': df.more_info['year'],
    'month': df.more_info['month'],
    'day': df.more_info['day'],
    'location': df.more_info['location'].decode('utf-8'),
    'systemID': df.more_info['systemID'],
    'time': df.data['time'][area.masktime][i],  # time as hour
    'range': area_range[max_i],  # range
    'depo': area_value[max_i],  # depo value
    'depo_1': area_value[max_i - 1],
    'depo_2': area_value[max_i - 2],
    'co_signal': area_snr[max_i],  # snr
    'co_signal1': area_snr[max_i-1],
    'co_signal2': area_snr[max_i-2],  # snr
    'vraw': area_vraw[max_i],  # v_raw
    'beta_raw': area_betaraw[max_i],  # beta_raw
    'cross_signal': area_cross[max_i]  # cross_signal
}])

# sub folder for each date
depo_sub_folder = depo_folder + '/' + df.filename
Path(depo_sub_folder).mkdir(parents=True, exist_ok=True)

# Append to or create new csv file
with open(depo_sub_folder + '/' + df.filename + '_depo.csv', 'a') as f:
    result.to_csv(f, header=f.tell() == 0, index=False)
# save fig
fig.savefig(depo_sub_folder + '/' + df.filename + '_' +
            str(int(df.data['time'][area.masktime][i]*1000)) + '.png')
