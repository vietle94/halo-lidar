import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import glob
import seaborn as sns
import numpy as np
from pathlib import Path
import matplotlib.dates as dates

%matplotlib qt

####################################
# Background SNR
####################################
# %% Define csv directory path
for site in ['32', '33', '46', '53', '54']:
    csv_path = 'F:/halo/' + site + '/depolarization/snr'

    # Collect csv file in csv directory
    data_list = glob.glob(csv_path + '/*_noise.csv')

    noise = pd.concat([pd.read_csv(f) for f in data_list],
                      ignore_index=True)
    noise = noise.astype({'year': int, 'month': int, 'day': int})
    noise['time'] = pd.to_datetime(noise[['year', 'month', 'day']]).dt.date
    name = noise['location'][0] + '-' + str(int(noise['systemID'][0]))
    # Calculate metrics
    groupby = noise.groupby('time')['noise']
    sd = groupby.std()
    if site == '32':
        sd[sd.index < pd.to_datetime('2017-11-22')].to_csv(
            'F:/halo/paper/' + name + '.csv')
        sd[sd.index >= pd.to_datetime('2017-11-22')].to_csv(
            'F:/halo/paper/' + 'Uto-32XR' + '.csv')
    else:
        continue
        sd.to_csv('F:/halo/paper/snr_background/' + name + '.csv')


# %%
site_names = ['Uto-32', 'Uto-32XR', 'Hyytiala-33',
              'Hyytiala-46', 'Vehmasmaki-53', 'Sodankyla-54']
fig, axes = plt.subplots(3, 2, figsize=(9, 6), sharex=True, sharey=True)
data_list = glob.glob('F:/halo/paper/snr_background/*.csv')
data_list = ['F:/halo/paper/snr_background\\Uto-32.csv',
             'F:/halo/paper/snr_background\\Uto-32XR.csv',
             'F:/halo/paper/snr_background\\Hyytiala-33.csv',
             'F:/halo/paper/snr_background\\Hyytiala-46.csv',
             'F:/halo/paper/snr_background\\Vehmasmaki-53.csv',
             'F:/halo/paper/snr_background\\Sodankyla-54.csv']
for file, ax, name in zip(data_list, axes.flatten(), site_names):
    sd = pd.read_csv(file)
    sd['time'] = pd.to_datetime(sd['time'])
    ax.scatter(sd['time'], sd['noise'], s=0.3)
    ax.set_title(name, weight='bold')
    ax.set_ylim([0, 0.0025])

    ax.xaxis.set_major_locator(dates.MonthLocator(6))
for ax in axes.flatten()[0::2]:
    ax.set_ylabel('Standard deviation')
fig.subplots_adjust(hspace=0.5)
fig.savefig('F:/halo/paper/figures/background_snr.png', dpi=150,
            bbox_inches='tight')

####################################
# Depo cloudbase
####################################

# %%
site_names = ['Uto-32', 'Uto-32XR', 'Hyytiala-33',
              'Hyytiala-46', 'Vehmasmaki-53', 'Sodankyla-54']
fig, axes = plt.subplots(3, 2, figsize=(9, 6), sharex=True)
df = pd.read_csv('F:/halo/paper/depo_cloudbase/result.csv')
for (systemID, group), ax, name in zip(df.groupby('systemID'),
                                       axes.flatten()[1:],
                                       site_names[1:]):
    if systemID == 32:
        axes[0, 0].hist(group['depo'][(group['depo'] > -0.01) & (group['depo'] < 0.1) & (pd.to_datetime(
            group[['year', 'month', 'day']]) < '2017-11-22')], bins=20)
        axes[0, 0].set_title('Uto-32', weight='bold')

        axes[0, 1].hist(group['depo'][(group['depo'] > -0.01) & (group['depo'] < 0.1) & (pd.to_datetime(
            group[['year', 'month', 'day']]) >= '2017-11-22')], bins=20)
        axes[0, 1].set_title('Uto-32XR', weight='bold')
    else:
        ax.hist(group['depo'][(group['depo'] > -0.01) & (group['depo'] < 0.1)], bins=20)
        ax.set_title(name, weight='bold')

axes.flatten()[-2].set_xlabel('Depolarization ratio')
axes.flatten()[-1].set_xlabel('Depolarization ratio')
fig.subplots_adjust(hspace=0.5)
fig.savefig('F:/halo/paper/figures/depo_cloudbase.png', dpi=150,
            bbox_inches='tight')

# %%
