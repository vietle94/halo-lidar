from datetime import datetime
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
result_path = 'F:\\halo\\summary'
Path(result_path).mkdir(parents=True, exist_ok=True)
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
ax.set_title('Integration time vs depolarization', weight='bold', size=22)
fig.savefig(result_path + '/integration_time.png',
            bbox_inches='tight', dpi=150)

# %%
fig, ax = plt.subplots(figsize=(12, 6))
sns.stripplot('integration_time', 'depo', hue='systemID',
              data=temp.reset_index(), ax=ax, alpha=0.5)
ax.set_title('Integration time vs depolarization', weight='bold', size=22)
fig.savefig(result_path + '/integration_time1.png',
            bbox_inches='tight', dpi=150)

# %%
depo['sys'] = depo['location'] + '-' + depo['systemID']
n_clusters = {'Kumpula-34': 2, 'Uto-32': 2, 'Uto-32XR': 1, 'Hyytiala-33': 1,
              'Hyytiala-46': 2, 'Vehmasmaki-53': 1, 'Sodankyla-54': 1,
              'Kumpula-146': 1}
# %%
for key, group in depo.groupby('sys'):
    fig2 = sns.relplot(x='time', y='depo',
                       col='month', data=group,
                       alpha=0.2,
                       linewidth=0, col_wrap=3, height=4.5, aspect=4/3)
    fig2.fig.savefig(result_path + '/' + key + '_depo_diurnal.png',
                     bbox_inches='tight', dpi=150)

    # Time series of other variables
    fig3, axes = plt.subplots(3, 2, figsize=(18, 9), sharex=True)
    for val, ax in zip(['co_signal', 'range', 'depo',
                        'vraw', 'beta_raw', 'cross_signal'],
                       axes.flatten()):
        ax.plot(group.groupby('date')[val].mean() if val != 'beta_raw' else
                np.log10(group.groupby('date')[val].mean()), '.')
        ax.set_title(val)
    fig3.suptitle('Mean values at cloud base with max SNR ' + key,
                  size=22, weight='bold')
    fig3.savefig(result_path + '/' + key + '_depo_other_vars.png',
                 bbox_inches='tight', dpi=150)

    # Pair plots
    fig4 = sns.pairplot(group, vars=['range', 'co_signal',
                                     'cross_signal', 'vraw',
                                     'beta_raw', 'depo'],
                        height=4.5, aspect=4/3)
    fig4.savefig(result_path + '/' + key + '_pairplot.png',
                 bbox_inches='tight', dpi=150)

    # Co-cross histogram
    co_cross_data = group[['co_signal', 'cross_signal']].dropna()
    H, co_edges, cross_edges = np.histogram2d(
        co_cross_data['co_signal'] - 1,
        co_cross_data['cross_signal'] - 1,
        bins=500)
    X, Y = np.meshgrid(co_edges, cross_edges)
    fig5, ax = plt.subplots(figsize=(18, 9))
    p = ax.pcolormesh(X, Y, H.T, norm=LogNorm())
    ax.set_xlabel('co_signal - 1')
    ax.set_ylabel('cross_signal - 1')
    colorbar = fig5.colorbar(p, ax=ax)
    colorbar.ax.set_ylabel('Number of observations')
    colorbar.ax.yaxis.set_label_position('left')
    ax.set_title('2D histogram of cross_signal vs co_signal ' + key,
                 size=22, weight='bold')
    ax.plot(co_cross_data['co_signal'] - 1,
            (co_cross_data['co_signal'] - 1) * 0.1, label='depo 0.1 fit',
            linewidth=0.5)
    ax.plot(co_cross_data['co_signal'] - 1,
            (co_cross_data['co_signal'] - 1) * 0.05, label='depo 0.05 fit',
            linewidth=0.5)
    ax.legend(loc='upper left')
    fig5.savefig(result_path + '/' + key + '_cross_vs_co.png',
                 bbox_inches='tight', dpi=200)

    # Histogram of depo
    temp = group['depo']
    if key == 'Kumpula-146':
        mask = (temp < 0.4) & (temp > -0.05)
    else:
        mask = (temp < 0.2) & (temp > -0.05)
    fig6, ax = plt.subplots(figsize=(18, 9))
    temp.loc[mask].hist(bins=50)
    ax.set_xlabel('Depo')

    if n_clusters[key] == 2:
        gmm = GaussianMixture(n_components=2, max_iter=1000)
        gmm.fit(temp[mask].values.reshape(-1, 1))
        smean = gmm.means_.ravel()
        sstd = np.sqrt(gmm.covariances_).ravel()
        sort_idx = np.argsort(smean)
        smean = smean[sort_idx]
        sstd = sstd[sort_idx]
        ax.set_title('Distribution of depo at cloud base, ' + key + f'\n\
        left peak is {smean[0]:.4f} $\pm$ {sstd[0]:.4f}', weight='bold')
    else:
        mean = np.mean(temp)
        std = np.std(temp)
        ax.set_title('Distribution of depo at cloud base, ' + key + f'\n\
            peak is {mean:.4f} $\pm$ {std:.4f}', weight='bold')

    fig6.savefig(result_path + '/' + key + '_depo_hist.png',
                 bbox_inches='tight', dpi=200)

    fig7, ax = plt.subplots(figsize=(12, 6))
    group_ = group.groupby('date').depo
    ax.errorbar(group['date'].unique(), group_.mean(), yerr=group_.std(),
                ls='none', marker='.', linewidth=0.5, markersize=5)
    ax.set_title('Mean value of depo at cloud base ' + key, weight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Depo')
    fig7.savefig(result_path + '/' + key + '_depo_scatter.png',
                 bbox_inches='tight', dpi=200)

    fig8, ax = plt.subplots(figsize=(12, 6))
    ax.plot(group['depo'], group['range'], '.')
    ax.set_title('Depolarization ratio vs range at ' + key, weight='bold')
    ax.set_xlabel('Depo')
    ax.set_ylabel('Range')
    fig8.savefig(result_path + '/' + key + '_depo_range.png',
                 bbox_inches='tight', dpi=200)
    plt.close('all')

# %%
weather_path = glob.glob(r'F:\weather/*.csv')

weather_list = []
for file in weather_path:
    weather_location = file.split('\\')[-1].split('_')[0]
    weather_file = pd.read_csv(file)
    weather_file['location'] = weather_location
    weather_list.append(weather_file)

weather = pd.concat(weather_list, ignore_index=True)
weather['location'] = weather['location'].str.title()
weather['location'] = ['Vehmasmaki' if x == 'Kuopio' else x
                       for x in weather['location']]

# %%
weather = weather.rename(columns={'Vuosi': 'year', 'Kk': 'month',
                                  'Pv': 'day', 'Klo': 'time'})
weather[['year', 'month', 'day']] = weather[['year',
                                             'month', 'day']].astype(str)
weather['month'] = weather['month'].str.zfill(2)
weather['day'] = weather['day'].str.zfill(2)
weather['datetime'] = weather['year'] + weather['month'] + \
    weather['day'] + weather['time']
weather['datetime'] = pd.to_datetime(weather['datetime'], format='%Y%m%d%H:%M')

# %%
depo['hour'] = depo['time']
depo['datetime'] = pd.to_datetime(depo[['year', 'month', 'day',
                                        'hour']])
depo['datetime_min10'] = depo.datetime.dt.round('10min')

# %%
depo_weather = depo.merge(weather, how='left',
                          left_on=['datetime_min10', 'location'],
                          right_on=['datetime', 'location'])

# Impute missing weather data
depo_weather.loc[depo_weather['datetime_y'].isnull(),
                 ['Pilvien määrä (1/8)',
                  'Suhteellinen kosteus (%)',
                  'Sateen intensiteetti (mm/h)']] = \
    weather.loc[
        (weather['datetime'] == pd.to_datetime('2018-06-07 07:10:00')) &
        (weather['location'] == 'Kumpula'),
        ['Pilvien määrä (1/8)',
         'Suhteellinen kosteus (%)',
         'Sateen intensiteetti (mm/h)']].values[0]


# %%
grp = depo_weather.groupby('sys')

fig, axes = plt.subplots(4, 2, figsize=(16, 9), sharex=True)
for (key, val), ax in zip(grp, axes.flatten()):
    val.plot('Suhteellinen kosteus (%)', 'depo',
             kind='scatter', ax=ax, title=key)
fig.subplots_adjust(hspace=0.3)
fig.suptitle('Relative humidity vs depo', weight='bold', size=22)
