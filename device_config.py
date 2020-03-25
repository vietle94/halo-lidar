import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import glob
import seaborn as sns
import numpy as np
import halo_data as hd
%matplotlib qt

# %%
# Specify data folder path
data_folder = r'F:\halo\32\depolarization'

# %%
# Get a list of all files in data folder
data = hd.getdata(data_folder)

# %%
# Choose a pattern that is in the file name
pattern = '2016'

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
# %%
device_config.describe()

# %%
device_config_varied = device_config.loc[:, (device_config != device_config.iloc[0]).any()]
settings = [i for i in device_config_varied.columns if i not in ['day', 'month', 'year']]
fig, axes = plt.subplots(len(settings), 1, figsize=(18, 9), sharex=True)
for ax, val in zip(axes.flatten(), settings):
    ax.plot(device_config_varied[val])
    ax.set_title(val)
fig.suptitle('Changes in device config ' + pattern,
             weight='bold',
             size=22)
