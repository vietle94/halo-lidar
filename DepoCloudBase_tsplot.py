import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns

# %%
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
fig, ax = plt.subplots(figsize=(18, 9))
sns.boxplot('date', 'depo', data=depo, ax=ax)
ax.set_title('Depo at cloud base time series', fontweight='bold')
ax.set_xlabel('date')
ax.set_ylabel('Depo')
