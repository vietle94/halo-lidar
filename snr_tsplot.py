import pandas as pd
import matplotlib.pyplot as plt
import glob
import seaborn as sns
import numpy as np

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
fig, ax = plt.subplots(2, 1, figsize=(18, 9), sharex=True)
noise.groupby('time')['noise'].std().plot(ax=ax[0])
# sns.boxplot('time', 'noise_sd', data=noise, ax=ax[0])
ax[0].set_title('SNR time series', fontweight='bold')
ax[0].set_xlabel('time')
ax[0].set_ylabel('SNR')

noise_avg.groupby('time')['noise'].std().plot(ax=ax[1])
ax[1].set_title('SNR in averaged data time series', fontweight='bold')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('SNR')
