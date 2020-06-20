import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns
from pathlib import Path
from matplotlib.colors import LogNorm
from sklearn.mixture import GaussianMixture

# %%
# Define csv directory path
depo_paths = [
    'F:\\halo\\32\\depolarization\\depo',
    'F:\\halo\\33\\depolarization\\depo',
    'F:\\halo\\34\\depo',
    'F:\\halo\\46\\depolarization\\depo',
    'F:\\halo\\53\\depolarization\\depo',
    'F:\\halo\\54\\depolarization\\depo',
    'F:\\halo\\146\\depolarization\\depo']
# Create saving folder
depo_result = 'F:\\halo\\summary'
Path(depo_result).mkdir(parents=True, exist_ok=True)
# Collect csv file in csv directory and subdirectory
result_list = []
for csv_path in depo_paths:
    data_list = [file
                 for path, subdir, files in os.walk(csv_path)
                 for file in glob.glob(os.path.join(path, '*.csv'))]
    result_list.extend(data_list)

# %%
depo = pd.concat([pd.read_csv(f) for f in result_list],
                 ignore_index=True)
depo = depo.astype({'year': int, 'month': int, 'day': int, 'systemID': int})
depo = depo.astype({'systemID': str})
# For right now, just take the date, ignore hh:mm:ss
depo['date'] = pd.to_datetime(depo[['year', 'month', 'day']])
depo.drop(['depo_1', 'co_signal1'], axis=1, inplace=True)
depo.loc[(depo['date'] > '2017-11-20') & (depo['systemID'] == '32'),
         'systemID'] = '32XR'

# %%
device_config_path = glob.glob(r'F:\halo\summary_device_config/*.csv')
device_config_df = []
for f in device_config_path:
    df = pd.read_csv(f)
    location_sysID = f.split('_')[-1].split('.')[0].split('-')
    df['location'] = location_sysID[0]
    df['systemID'] = location_sysID[1]
    df.rename(columns={'time': 'date'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    device_config_df.append(df)

# %%
device_config = pd.concat(device_config_df, ignore_index=True)
device_config.loc[(device_config['date'] > '2017-11-20') &
                  (device_config['systemID'] == '32'),
                  'systemID'] = '32XR'
device_config.loc[device_config['systemID'].isin(['32XR', '146']),
                  'prf'] = 10000
device_config['integration_time'] = \
    device_config['num_pulses_m1'] / device_config['prf']

# %%
result = depo.merge(device_config, on=['systemID', 'date'])
temp = result.groupby(['systemID', 'date']).mean()

# %%
fig, ax = plt.subplots(figsize=(12, 6))
sns.scatterplot('integration_time', 'depo', hue='systemID',
                data=temp.reset_index(), ax=ax, alpha=0.2)

# %%
fig, ax = plt.subplots(figsize=(12, 6))
sns.stripplot('integration_time', 'depo', hue='systemID',
              data=temp.reset_index(), ax=ax, alpha=0.5)

# %%
fig2 = sns.relplot(x='time', y='depo',
                    col='month', data=depo,
                    alpha=0.2,
                    linewidth=0, col_wrap=3, height=4.5, aspect=4/3)
 fig2.fig.savefig(depo_result + '/' + name + '_depo_diurnal.png')

  # %%
  fig4, axes = plt.subplots(3, 2, figsize=(18, 9), sharex=True)
   for val, ax in zip(['co_signal', 'range', 'depo',
                        'vraw', 'beta_raw', 'cross_signal'],
                       axes.flatten()):
        ax.plot(depo.groupby('date')[val].mean() if val is not 'beta_raw' else np.log10(
            depo.groupby('date')[val].mean()), '.')
        ax.set_title(val)
    fig4.suptitle('Mean values at cloud base with max SNR ' + name,
                  size=22, weight='bold')
    fig4.savefig(depo_result + '/' + name + '_depo_other_vars.png')

    # %%
    fig7 = sns.pairplot(depo, vars=['range', 'co_signal', 'cross_signal', 'vraw',
                                             'beta_raw', 'depo'],
                        height=4.5, aspect=4/3)
    fig7.savefig(depo_result + '/' + name + '_pairplot.png')

    # %%
    co_cross_data = depo[['co_signal', 'cross_signal']]
    co_cross_data.dropna(inplace=True)
    H, co_edges, cross_edges = np.histogram2d(co_cross_data['co_signal'] - 1,
                                              co_cross_data['cross_signal'] - 1,
                                              bins=500)
    X, Y = np.meshgrid(co_edges, cross_edges)
    fig8, ax = plt.subplots(figsize=(18, 9))
    p = ax.pcolormesh(X, Y, H.T, norm=LogNorm())
    ax.set_xlabel('co_signal - 1')
    ax.set_ylabel('cross_signal - 1')
    colorbar = fig8.colorbar(p, ax=ax)
    colorbar.ax.set_ylabel('Number of observations')
    colorbar.ax.yaxis.set_label_position('left')
    ax.set_title('2D histogram of cross_signal vs co_signal ' + name,
                 size=22, weight='bold')
    ax.plot(co_cross_data['co_signal'] - 1,
            (co_cross_data['co_signal'] - 1) * 0.1, label='depo 0.1 fit',
            linewidth=0.5)
    # ax.plot(co_cross_data['co_signal'] - 1,
    #         (co_cross_data['co_signal'] - 1) * 0.07, label='depo 0.07 fit',
    #         linewidth=0.5)
    ax.legend(loc='upper left')
    fig8.savefig(depo_result + '/' + name + '_cross_vs_co.png',
                 bbox_inches='tight', dpi=200)

    # %%
    # For bimodal distribution
    # temp = depo.loc[depo['date'] < pd.to_datetime('2017-09-01'), 'depo']
    temp = depo['depo']
    mask = (temp < 0.2) & (temp > -0.05)
    fig0, ax = plt.subplots(figsize=(18, 9))
    temp.loc[mask].hist(bins=50)
    ax.set_xlabel('Depo')

    gmm = GaussianMixture(n_components=2, max_iter=1000)
    gmm.fit(temp[mask].values.reshape(-1, 1))
    smean = gmm.means_.ravel()
    sstd = np.sqrt(gmm.covariances_).ravel()
    sort_idx = np.argsort(smean)
    smean = smean[sort_idx]
    sstd = sstd[sort_idx]

    ax.set_title('Distribution of depo at cloud base, ' + name + f'\n\
    left peak is {smean[0]:.4f} $\pm$ {sstd[0]:.4f}', weight='bold')
    fig0.savefig(depo_result + '/' + name + '_depo_hist.png',
                 bbox_inches='tight', dpi=200)

    # %%
    # temp = depo.loc[depo['date'] > pd.to_datetime('2017-09-01'), 'depo']
    temp = depo['depo']
    mask = (temp < 0.2) & (temp > -0.05)
    fig0, ax = plt.subplots(figsize=(18, 9))
    temp.loc[mask].hist(bins=50)
    ax.set_xlabel('Depo')
    mean = np.mean(temp)
    std = np.std(temp)
    ax.set_title('Distribution of depo at cloud base, ' + name + f'\n\
    peak is {mean:.4f} $\pm$ {std:.4f}', weight='bold')
    fig0.savefig(depo_result + '/' + name + '_depo_hist.png',
                 bbox_inches='tight', dpi=200)

    # %%
    fig, ax = plt.subplots(figsize=(12, 6))
    group = depo.groupby('date').depo
    ax.errorbar(depo['date'].unique(), group.mean(), yerr=group.std(),
                ls='none', marker='.', linewidth=0.5, markersize=5)
    ax.set_title('Mean value of depo at cloud base ' + name, weight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Depo')
    fig.savefig(depo_result + '/' + name + '_depo_scatter.png',
                bbox_inches='tight', dpi=200)

# %%
a = [1, 2, 4]
b = [3, 4, 4]

a.extend(b)
a
