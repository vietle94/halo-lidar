import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns

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
# Boxplot for all depo types
fig, ax = plt.subplots(2, 1, figsize=(18, 9), sharex=True)
sns.boxplot('date', 'value', data=depo, ax=ax[0])
ax[0].set_title('Depo at cloud base time series', fontweight='bold')
ax[0].set_xlabel('date')
ax[0].set_ylabel('Depo')
ax[0].tick_params(axis='x', labelrotation=45)

sns.boxplot('date', 'value', data=depo[depo['depo_type'] == 'depo'], ax=ax[1])
ax[1].set_title('Depo at cloud base time series at max SNR', fontweight='bold')
ax[1].set_xlabel('date')
ax[1].set_ylabel('Depo')
ax[1].tick_params(axis='x', labelrotation=45)

# %%
fig, ax = plt.subplots(2, 1, figsize=(18, 9), sharex=True)
sns.boxplot('date', 'value', data=depo[depo['depo_type'] == 'depo_1'], ax=ax[0])
ax[0].set_title('Depo at cloud base time series at 1 level below max SNR', fontweight='bold')
ax[0].set_xlabel('date')
ax[0].set_ylabel('Depo')
ax[0].tick_params(axis='x', labelrotation=45)

sns.boxplot('date', 'value', data=depo[depo['depo_type'] == 'depo_2'], ax=ax[1])
ax[1].set_title('Depo at cloud base time series at 2 levels below max SNR', fontweight='bold')
ax[1].set_xlabel('date')
ax[1].set_ylabel('Depo')
ax[1].tick_params(axis='x', labelrotation=45)

# %%
fig, ax = plt.subplots(figsize=(18, 9))
sns.boxplot('date', 'value', hue='depo_type', data=depo, ax=ax)
ax.set_title('Depo at cloud base time series', fontweight='bold')
ax.set_xlabel('date')
ax.set_ylabel('Depo')
ax.tick_params(axis='x', labelrotation=45)

# %%
fig, ax = plt.subplots(figsize=(18, 9))
sns.lineplot('time', 'value', hue='date', style='depo_type', data=depo, ax=ax)
ax.set_title('Depo at cloud base time series', fontweight='bold')
ax.tick_params(axis='x', labelrotation=45)

# %%
fig, ax = plt.subplots(figsize=(18, 9))
sns.scatterplot('time', 'value', hue='depo_type', data=depo, ax=ax)
ax.set_title('Depo at cloud base time series', fontweight='bold')
ax.tick_params(axis='x', labelrotation=45)
