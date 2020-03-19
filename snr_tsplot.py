import pandas as pd
import matplotlib.pyplot as plt
import glob
import seaborn as sns
import numpy as np
%matplotlib qt
# %% Define csv directory path
csv_path = r'F:\halo\32\depolarization\snr'

# Collect csv file in csv directory
data_list = glob.glob(csv_path + '/*_noise.csv')
data_avg_list = glob.glob(csv_path + '/*noise_avg.csv')

# %%
noise = pd.concat([pd.read_csv(f) for f in data_list],
                  ignore_index=True)
noise = noise.astype({'year': int, 'month': int, 'day': int})
noise['time'] = pd.to_datetime(noise[['year', 'month', 'day']]).dt.date

noise_avg = pd.concat([pd.read_csv(f) for f in data_avg_list],
                      ignore_index=True)
noise_avg = noise_avg.astype({'year': int, 'month': int, 'day': int})
noise_avg['time'] = pd.to_datetime(noise_avg[['year', 'month', 'day']]).dt.date

# %%
# pick only sd
fig, ax = plt.subplots(figsize=(18, 9))
noise.groupby('time')['noise'].std().plot(ax=ax)
noise_avg.groupby('time')['noise'].std().plot(ax=ax, label='noise in avg data')
ax.set_title('SNR time series', fontweight='bold')
ax.set_xlabel('time')
ax.set_ylabel('SNR')
ax.legend()

# %%
fig, ax = plt.subplots(figsize=(18, 9))
sns.boxplot('time', 'noise', data=noise, ax=ax)
ax.tick_params(axis='x', labelrotation=45)
