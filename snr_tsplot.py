import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import glob
import seaborn as sns
import numpy as np
from pathlib import Path
%matplotlib qt

# %% Define csv directory path
csv_path = r'F:\halo\32\depolarization\snr'
# Create saving folder
snr_result = csv_path + '/result'
Path(snr_result).mkdir(parents=True, exist_ok=True)

# Collect csv file in csv directory
data_list = glob.glob(csv_path + '/*_noise.csv')
# data_avg_list = glob.glob(csv_path + '/*noise_avg.csv')

# %%
noise = pd.concat([pd.read_csv(f) for f in data_list],
                  ignore_index=True)
noise = noise.astype({'year': int, 'month': int, 'day': int})
noise['time'] = pd.to_datetime(noise[['year', 'month', 'day']]).dt.date

# noise_avg = pd.concat([pd.read_csv(f) for f in data_avg_list],
#                       ignore_index=True)
# noise_avg = noise_avg.astype({'year': int, 'month': int, 'day': int})
# noise_avg['time'] = pd.to_datetime(noise_avg[['year', 'month', 'day']]).dt.date

# %%
# Calculate metrics
groupby = noise.groupby('time')['noise']
sd = groupby.std()
mean = groupby.mean()
# q25, q75 = groupby.quantile(0.25), groupby.quantile(0.75)

# groupby_avg = noise_avg.groupby('time')['noise']
# sd_avg = groupby_avg.std()
# mean_avg = groupby_avg.mean()
# q25_avg, q75_avg = groupby_avg.quantile(0.25), groupby_avg.quantile(0.75)

# %%
# pick only sd
# fig1, ax = plt.subplots(2, 1, figsize=(18, 9))
# sd.plot(ax=ax[0], label='Normal data')
# sd_avg.plot(ax=ax[0], label='Averaged data')
# ax[0].set_title('Standard deviation of noise time series', fontweight='bold')
# ax[0].set_ylabel('SNR')
# ax[0].legend()
#
# mean.plot(ax=ax[1], label='Normal data')
# mean_avg.plot(ax=ax[1], alpha=0.3, label='Averaged data')
# ax[1].set_title('Mean of noise time series', fontweight='bold')
# ax[1].axhline(0, linestyle='--', c='gray')
# ax[1].legend()
# fig1.savefig(snr_result + '/noise_sd.png')
# fig1.subplots_adjust(hspace=0.3)
# %%
# fig2, ax = plt.subplots(figsize=(18, 9))
# ax.plot(mean, color='orange', label='noise mean')
# ax.fill_between(mean.index, q25, q75, alpha=.1, color='orange')
# ax.plot(mean_avg, color='#1f77b4', label='noise mean in avg data')
# ax.fill_between(mean_avg.index, q25_avg, q75_avg, alpha=.1, color='#1f77b4')
# ax.legend()
# ax.set_title('Noise mean level in 25 and 50 percentiles')
# fig2.savefig(snr_result + '/noise_ts.png')

# %%
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(sd, '.')
ax.set_title('Standard deviation of SNR time series', fontweight='bold')
ax.set_ylabel('Standard deviation of SNR')
ax.set_xlabel('Time')
fig.savefig(snr_result + '/noise_sd_report.png',
            bbox_inches='tight', dpi=150)
