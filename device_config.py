import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import glob
import seaborn as sns
import numpy as np
import halo_data as hd
from pathlib import Path
%matplotlib qt

# %%
# Specify data folder path
data_folder = r'F:\halo\34\depolarization'
# Folder to save
save_folder = r'F:\halo\34'
# %%
# Get a list of all files in data folder
data = hd.getdata(data_folder)

# %%
# Choose a pattern that is in the file name, empty string means taking all files
pattern = ''

# %%
device_config = pd.DataFrame()
for f in data:
    if pattern in f.replace(data_folder, ''):
        df = hd.halo_data(f)
        result = {}
        result.update(df.info)
        result.update({'day': df.more_info['day'],
                       'month': df.more_info['month'],
                       'year': df.more_info['year']})
        device_config = device_config.append(pd.DataFrame.from_dict([result]),
                                             ignore_index=True)
device_config = device_config.astype({'year': int, 'month': int, 'day': int})
device_config['time'] = pd.to_datetime(device_config[['year', 'month', 'day']]).dt.date
device_config = device_config.set_index('time')

# Extract name for this location from the last file
location_name = df.more_info['location'] + '-' + str(int(df.more_info['systemID']))

# %%
device_config.describe()

# %%
# Extract only non-constant variables
device_config_varied = device_config.loc[:, (device_config != device_config.iloc[0]).any()]
settings = [i for i in device_config_varied.columns if i not in ['day', 'month', 'year']]

# %%
fig, axes = plt.subplots(len(settings), 1, figsize=(18, 9), sharex=True)
if len(settings) > 1:
    for ax, val in zip(axes, settings):
        ax.plot(device_config_varied[val])
        ax.set_title(val)
else:
    axes.plot(device_config_varied[settings])
    axes.set_title(settings)
fig.suptitle('Changes in device config ' + pattern + location_name,
             weight='bold',
             size=22)
fig.subplots_adjust(hspace=0.4)

# %%
# Save figure and csv file
device_config_folder = save_folder + '/device_config'
Path(device_config_folder).mkdir(parents=True, exist_ok=True)
fig.savefig(device_config_folder + '/device_config_' + location_name + '.png')
device_config.to_csv(device_config_folder + '/device_config_' + location_name + '.csv')
