import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns
import matplotlib.ticker as ticker
import calendar

%matplotlib qt
# %% Define csv directory path
csv_path = r'F:\halo\32\depolarization\depo'

# Collect csv file in csv directory and subdirectory
data_list = all_csv_files = [file
                             for path, subdir, files in os.walk(csv_path)
                             for file in glob.glob(os.path.join(path, '*.csv'))]

# %%
depo = pd.concat([pd.read_csv(f) for f in data_list],
                 ignore_index=True)
depo = depo.astype({'year': int, 'month': int, 'day': int})

# For right now, just take the date, ignore hh:mm:ss
depo['date'] = pd.to_datetime(depo[['year', 'month', 'day']]).dt.date

# %%

depo = pd.melt(depo, id_vars=[x for x in depo.columns if 'depo' not in x],
               value_vars=[x for x in depo.columns if 'depo' in x],
               var_name='depo_type')

# %%
datelabel = depo.date.unique()
fig, ax = plt.subplots(figsize=(18, 9))
sns.boxplot('date', 'value', hue='depo_type', data=depo, ax=ax)
ax.set_title('Depo at cloud base time series', fontweight='bold')
ax.set_xlabel('2016', weight='bold')
ax.set_ylabel('Depo')
# ax.tick_params(axis='x', labelrotation=45)
# Space out interval for xticks
ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(0, len(datelabel), 5)))
ax.xaxis.set_major_formatter(ticker.FixedFormatter([x.strftime("%b %d") for x in datelabel][0::5]))

# %%
sns.relplot(x='time', y='value',
            col='month', data=depo[depo['depo_type'] == 'depo'], alpha=0.5,
            col_wrap=3)
# %%
fig, ax = plt.subplots(figsize=(18, 9))
depo.groupby('depo_type')['value'].hist(bins=50, ax=ax, alpha=0.5)
ax.legend(['depo', 'depo_1'])
ax.set_title('Distribution of depo at max SNR and 1 level below')
ax.set_xlabel('Depo')

# %%
# fig, ax = plt.subplots(figsize=(18, 9))
# for name, group in depo.groupby('depo_type'):
#     ax.plot(group.groupby('date').value.mean(), '.', label=name)
# ax.legend()
# ax.set_title('Mean value of depo at max SNR and 1 level below')
# ax.set_xlabel('Date')
# ax.set_ylabel('Depo')

# %%
fig, axes = plt.subplots(3, 2, figsize=(18, 9), sharex=True)
for val, ax in zip(['co_signal', 'co_signal1', 'range',
                    'vraw', 'beta_raw', 'cross_signal'],
                   axes.flatten()):
    ax.plot(depo.groupby('date')[val].mean() if val is not 'beta_raw' else np.log10(
        depo.groupby('date')[val].mean()), '.')
    ax.set_title(val)
fig.suptitle('Mean values of various metrics at cloud base with max SNR',
             size=22, weight='bold')
