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
image_folder = data_folder + r'\img'
# Specify folder path for output depo analysis
depo_folder = data_folder + r'\depo'
# Specify folder path for snr collection
snr_folder = data_folder + r'\snr'
# Make those specified folders if they are not existed yet
for path in [image_folder, depo_folder, snr_folder]:
    Path(path).mkdir(parents=True, exist_ok=True)

# %%
# Get a list of all files in data folder
data = hd.getdata(data_folder)

# %%
# pick date of data
pick_date = '20170102'
data_indices = hd.getdata_date(data, pick_date, data_folder)
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
print(df.filename)
# Change masking missing values from -999 to NaN
df.unmask999()
# Remove bottom 3 height layers
df.filter_height()
# Overview of data
df.describe()
# View meta data of all variables, uncomment following line and run to see
# df.meta_data

# %%
# Plot data and save it
%matplotlib qt
df.plot(variables=['beta_raw', 'v_raw', 'cross_signal', 'depo_raw', 'co_signal',
                   'cross_signal_averaged', 'depo_averaged_raw', 'co_signal_averaged'],
        ncol=2, size=(18, 9)).savefig(image_folder + '/' + df.filename + '_raw.png')

# %%
# Close the plot
plt.close()

# %%
# Choose background noise and calculate thresholds
df.snr_filter(multiplier=3, multiplier_avg=3)

# %%
# Close the plot
plt.close()

# %%
# Append to or create new csv file
df.snr_save(snr_folder)

# %%
# Use those obtained threshold to filter data, this will overwrite raw data
df.filter(variables=['beta_raw', 'v_raw', 'cross_signal', 'depo_raw'],
          ref='co_signal', threshold=df.area_snr.threshold)

df.filter(variables=['cross_signal_averaged', 'depo_averaged_raw'],
          ref='co_signal_averaged', threshold=df.area_snr_avg.threshold)

# %%
# Plot filtered data and save it
df.plot(variables=['beta_raw', 'v_raw', 'cross_signal', 'depo_raw', 'co_signal',
                   'cross_signal_averaged', 'depo_averaged_raw', 'co_signal_averaged'],
        ncol=2, size=(18, 9)).savefig(image_folder + '/' + df.filename + '_filtered.png')

# %%
##################################################
#
# STOP HERE if you don't like any clouds
#
##################################################
# Close the plot
plt.close()

# %% Summary after filtering
df.describe()

# %%
##################################################
#
# Area selection for depo cloud at each time point, press d to move forward
#
##################################################
fig_timeprofile = df.depo_timeprofile()

# %%
# Extract data from each time point, run this line multiple time for each
# time point that you want to save
df.depo_timeprofile_save(fig_timeprofile, depo_folder)

# %%
##################################################
#
# Area selection for whole cloud
#
##################################################
fig_wholeprofile = df.depo_wholeprofile()

# %%
# Extract data from whole cloud, run this line multiple time for each
# whole cloud that you want to save
df.depo_wholeprofile_save(fig_wholeprofile, depo_folder)
