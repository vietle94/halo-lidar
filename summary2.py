import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from matplotlib.colors import LogNorm
import matplotlib
import copy
import glob
%matplotlib qt

# %%
missing_df = pd.DataFrame({})
for site in ['46', '54', '33', '53', '34', '32']:
    path = 'F:\\halo\\classifier2\\' + site + '\\'
    list_files = glob.glob(path + '*.csv')
    time_df = pd.DataFrame(
        {'date': [file.split('\\')[-1][:10] for
                  file in list_files if 'result' not in file],
         'location2': [file.split('\\')[-1].split('-')[3] for
                       file in list_files if 'result' not in file]})
    time_df['date'] = pd.to_datetime(time_df['date'])
    time_df['year'] = time_df['date'].dt.year
    time_df['month'] = time_df['date'].dt.month
    time_df_count = time_df.groupby(['year', 'month', 'location2']).count()
    time_df_count = time_df_count.reset_index().rename(columns={'date': 'count'})
    missing_df = missing_df.append(time_df_count, ignore_index=True)
missing_df.loc[missing_df['location2'] == 'Kuopio', 'location2'] = 'Hyytiala'
missing_df = missing_df.set_index(['year', 'month', 'location2'])
mux = pd.MultiIndex.from_product([missing_df.index.levels[0],
                                  missing_df.index.levels[1],
                                  missing_df.index.levels[2]],
                                 names=['year', 'month', 'location2'])
missing_df = missing_df.reindex(mux, fill_value=0).reset_index()
missing_df = missing_df[missing_df['count'] < 15]

# %%
my_cmap = copy.copy(matplotlib.cm.get_cmap('jet'))
my_cmap.set_under('w')
bin_depo = np.linspace(0, 0.5, 50)
bin_month = np.arange(0.5, 13, 1)
bin_time = np.arange(0, 25)

# %%
df = pd.DataFrame()
for site in ['46', '54', '33', '53', '32']:
    save_location = 'F:\\halo\\classifier2\\summary\\'
    df = df.append(pd.read_csv('F:\\halo\\classifier2\\' + site + '\\result.csv'),
                   ignore_index=True)

df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['depo'][df['depo'] > 1] = np.nan
df.loc[df['location'] == 'Kuopio-33', 'location'] = 'Hyytiala-33'
df[['location2', 'ID']] = df['location'].str.split('-', expand=True)

# %%
df[(df['location2'] == 'Uto') &
    (df['date'] >= '2019-12-06') &
    (df['date'] <= '2019-12-10')] = np.nan

# %%
pos = {'Uto': 3, 'Hyytiala': 2,
       'Vehmasmaki': 1, 'Sodankyla': 0}
y = {2016: 0, 2017: 1, 2018: 2, 2019: 3}
cbar_max = {'Uto': 600, 'Hyytiala': 600,
            'Vehmasmaki': 400, 'Sodankyla': 400}
# cbar_max = {'Uto': 600, 'Hyytiala': 450,
#             'Vehmasmaki': 330, 'Sodankyla': 260}
avg = {}

X, Y = np.meshgrid(bin_month, bin_depo)

fig, axes = plt.subplots(4, 5, figsize=(18, 9), sharex=True, sharey=True)
for key, grp in df.groupby(['year']):
    for k, g in grp.groupby(['location2']):
        H, month_edge, depo_edge = np.histogram2d(
            g['month'], g['depo'],
            bins=(bin_month, bin_depo)
        )
        miss = missing_df[(missing_df['year'] == key) &
                          (missing_df['location2'] == k)]['month']
        if len(miss.index) != 0:
            for miss_month in miss:
                H[miss_month-1, :] = np.nan
        p = axes[pos[k], y[key]].pcolormesh(X, Y, H.T, cmap=my_cmap,
                                            vmin=0.1, vmax=cbar_max[k])
        fig.colorbar(p, ax=axes[pos[k], y[key]])
        axes[pos[k], y[key]].xaxis.set_ticks([4, 8, 12])

        if k not in avg:
            avg[k] = H[:, :, np.newaxis]
        else:
            avg[k] = np.append(avg[k], H[:, :, np.newaxis], axis=2)

for i, val in enumerate(y.keys()):
    axes[0, i].set_title(val, weight='bold', size=15)
    axes[3, i].set_xlabel('Month')
    axes[i, 0].set_ylabel('Depo')

for i, key in enumerate(pos.keys()):
    lab_ax = fig.add_subplot(414 - i, frameon=False)
    lab_ax.tick_params(labelcolor='none', top=False, bottom=False,
                       left=False, right=False)
    lab_ax.set_ylabel(key, weight='bold', labelpad=20, size=15)

for key, val in avg.items():

    p = axes[pos[key], -1].pcolormesh(X, Y, np.nanmean(val, axis=2).T,
                                      cmap=my_cmap,
                                      vmin=0.1, vmax=cbar_max[key])
    fig.colorbar(p, ax=axes[pos[key], -1])
    axes[pos[k], -1].xaxis.set_ticks([4, 8, 12])
axes[0, -1].set_title('4 years averaged', weight='bold', size=15)
fig.tight_layout()
fig.savefig(save_location + 'month_depo.png', bbox_inches='tight')

# %%
avg = {}
X, Y = np.meshgrid(bin_time, bin_month)
fig, axes = plt.subplots(4, 5, figsize=(18, 9), sharex=True, sharey=True)
for key, grp in df.groupby(['year']):
    for k, g in grp.groupby(['location2']):
        dep_mean, time_edge, month_edge, _ = binned_statistic_2d(
            g['time'],
            g['month'],
            g['depo'],
            bins=[bin_time, bin_month],
            statistic=np.nanmean)
        dep_count, _, _, _ = binned_statistic_2d(
            g['time'],
            g['month'],
            g['depo'],
            bins=[bin_time, bin_month],
            statistic='count')
        dep_mean[dep_count < 25] = np.nan
        p = axes[pos[k], y[key]].pcolormesh(X, Y, dep_mean.T, cmap='jet',
                                            vmin=1e-5, vmax=0.3)
        fig.colorbar(p, ax=axes[pos[k], y[key]])
        axes[pos[k], y[key]].yaxis.set_ticks([4, 8, 12])

        if k not in avg:
            avg[k] = dep_mean[:, :, np.newaxis]
        else:
            avg[k] = np.append(avg[k], dep_mean[:, :, np.newaxis], axis=2)

for i, val in enumerate(y.keys()):
    axes[0, i].set_title(val, weight='bold', size=15)
    axes[3, i].set_xlabel('Time (hour)')
    axes[i, 0].set_ylabel('Month')

for i, key in enumerate(pos.keys()):
    lab_ax = fig.add_subplot(414 - i, frameon=False)
    lab_ax.tick_params(labelcolor='none', top=False, bottom=False,
                       left=False, right=False)
    lab_ax.set_ylabel(key, weight='bold', labelpad=20, size=15)

for key, val in avg.items():
    p = axes[pos[key], -1].pcolormesh(X, Y, np.nanmean(val, axis=2).T,
                                      cmap='jet',
                                      vmin=1e-5, vmax=0.3)
    cbar = fig.colorbar(p, ax=axes[pos[key], -1])
    cbar.ax.set_ylabel('Depolarization ratio')

axes[0, -1].set_title('4 years averaged', weight='bold', size=15)
axes[-1, -1].set_xlabel('Time (hour)')
fig.tight_layout()
fig.savefig(save_location + 'month_time.png', bbox_inches='tight')

# %%
avg = {}
cbar_max2 = {'Uto': 300, 'Hyytiala': 400,
             'Vehmasmaki': 300, 'Sodankyla': 200}

X, Y = np.meshgrid(bin_time, bin_month)
fig, axes = plt.subplots(4, 5, figsize=(18, 9), sharex=True, sharey=True)
for key, grp in df.groupby(['year']):
    for k, g in grp.groupby(['location2']):
        dep_mean, time_edge, month_edge, _ = binned_statistic_2d(
            g['time'],
            g['month'],
            g['depo'],
            bins=[bin_time, bin_month],
            statistic='count')
        p = axes[pos[k], y[key]].pcolormesh(X, Y, dep_mean.T, cmap=my_cmap,
                                            vmin=0.001, vmax=cbar_max2[k])
        cbar = fig.colorbar(p, ax=axes[pos[k], y[key]])
        axes[pos[k], y[key]].yaxis.set_ticks([4, 8, 12])

        if k not in avg:
            avg[k] = dep_mean[:, :, np.newaxis]
        else:
            avg[k] = np.append(avg[k], dep_mean[:, :, np.newaxis], axis=2)

for i, val in enumerate(y.keys()):
    axes[0, i].set_title(val, weight='bold', size=15)
    axes[3, i].set_xlabel('Time (hour)')
    axes[i, 0].set_ylabel('Month')

for i, key in enumerate(pos.keys()):
    lab_ax = fig.add_subplot(414 - i, frameon=False)
    lab_ax.tick_params(labelcolor='none', top=False, bottom=False,
                       left=False, right=False)
    lab_ax.set_ylabel(key, weight='bold', labelpad=20, size=15)

for key, val in avg.items():
    p = axes[pos[key], -1].pcolormesh(X, Y, np.nanmean(val, axis=2).T,
                                      cmap=my_cmap,
                                      vmin=0.001, vmax=cbar_max2[key])
    fig.colorbar(p, ax=axes[pos[key], -1])
axes[0, -1].set_title('4 years averaged', weight='bold', size=15)
axes[-1, -1].set_xlabel('Time (hour)')
fig.tight_layout()
fig.savefig(save_location + 'month_time_count.png', bbox_inches='tight')

# %%
df2 = pd.read_csv('smeardata_20160101120000.csv')
df2['time'] = pd.to_datetime(df2[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']])
df2 = df2.drop(['HYY_META.SO21250'], axis=1)
df2 = df2.rename(columns={'HYY_META.CO1250': 'CO',
                          'HYY_META.O31250': 'O3'})

# %%
fig, axes = plt.subplots(2, 4, figsize=(16, 5),
                         sharex=True, sharey=True)
for year, grp_year in df2.groupby('Year'):
    CO_mean, time_edge, month_edge, _ = binned_statistic_2d(
        grp_year['Hour'],
        grp_year['Month'],
        grp_year['CO'],
        bins=[bin_time, bin_month],
        statistic=np.nanmean)

    X, Y = np.meshgrid(time_edge, month_edge)
    p = axes[0, y[year]].pcolormesh(X, Y, CO_mean.T, cmap='jet',
                                    vmin=100, vmax=200)
    cbar = fig.colorbar(p, ax=axes[0, y[year]])
    if year == 2019:
        cbar.ax.set_ylabel('CO concentration')

    O3_mean, time_edge, month_edge, _ = binned_statistic_2d(
        grp_year['Hour'],
        grp_year['Month'],
        grp_year['O3'],
        bins=[bin_time, bin_month],
        statistic=np.nanmean)

    X, Y = np.meshgrid(time_edge, month_edge)
    p = axes[1, y[year]].pcolormesh(X, Y, O3_mean.T, cmap='jet',
                                    vmin=20, vmax=50)
    cbar = fig.colorbar(p, ax=axes[1, y[year]])
    if year == 2019:
        cbar.ax.set_ylabel('O3 concentration')
    axes[1, y[year]].yaxis.set_ticks([4, 8, 12])
    axes[0, y[year]].set_title(year, weight='bold')
    axes[1, y[year]].set_xlabel('Time (hour)', weight='bold')
axes[0, 0].set_ylabel('Month')
axes[1, 0].set_ylabel('Month')
fig.suptitle('Auxiliary data at Hyytiala', weight='bold', size=22)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(save_location + 'auxiliary_hyytiala.png', bbox_inches='tight')

# %%
list_weather = glob.glob('F:/weather/*.csv')
location_weather = {'hyytiala': 'Hyytiala', 'kuopio': 'Vehmasmaki',
                    'sodankyla': 'Sodankyla', 'uto': 'Uto'}
weather = pd.DataFrame()
for file in list_weather:
    if 'kumpula' in file:
        continue
    df_file = pd.read_csv(file)
    df_file['location2'] = location_weather[file.split('\\')[-1].split('_')[0]]
    weather = weather.append(df_file, ignore_index=True)

weather = weather.rename(columns={'Vuosi': 'year', 'Kk': 'month',
                                  'Pv': 'day', 'Klo': 'time',
                                  'Suhteellinen kosteus (%)': 'RH',
                                  'Ilman lämpötila (degC)': 'Temperature'})
weather[['year', 'month', 'day']] = weather[['year',
                                             'month', 'day']].astype(str)
weather['month'] = weather['month'].str.zfill(2)
weather['day'] = weather['day'].str.zfill(2)
weather['datetime'] = weather['year'] + weather['month'] + \
    weather['day'] + weather['time']
weather['datetime'] = pd.to_datetime(weather['datetime'], format='%Y%m%d%H:%M')


df['hour'] = np.floor(df['time'])
df['minute'] = (df['time'] % 1) * 60
df['second'] = 0
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute', 'second']])
df = df.drop(['year', 'month', 'day', 'hour', 'minute', 'second'], axis=1)


weather = weather.set_index('datetime').resample('0.5H').mean()
weather = weather.reset_index()


df = pd.merge(weather, df)

# %%
bimonthly_ = [np.arange(1, 7).reshape(3, 2),
              np.arange(7, 13).reshape(3, 2)]
for ii, bimonthly in enumerate(bimonthly_):
    fig, ax = plt.subplots(3, 2, figsize=(16, 9), sharey=True)
    for i, month in enumerate(bimonthly):
        for loc, grp in df[(df.datetime.dt.month >= month[0]) & (df.datetime.dt.month <= month[1])].groupby('location2'):
            grp_avg = grp.groupby('range')['depo'].mean()
            grp_count = grp.groupby('range')['depo'].count()
            grp_std = grp.groupby('range')['depo'].std()
            grp_avg = grp_avg[grp_count > 0.01 * sum(grp_count)]
            grp_std = grp_std[grp_count > 0.01 * sum(grp_count)]
            ax[i, 0].errorbar(grp_avg.index + np.random.uniform(-30, 30),
                              grp_avg, grp_std, label=loc, fmt='.')
            ax[i, 0].set_xlim([-50, 2500])
            ax[i, 0].set_xlabel('Range')

            grp_ground = grp[grp['range'] <= 300]
            y_grp = grp_ground.groupby(pd.cut(grp_ground['RH'], np.arange(0, 110, 10)))
            y_mean = y_grp['depo'].mean()
            y_std = y_grp['depo'].std()
            x = np.arange(5, 105, 10)
            ax[i, 1].errorbar(x + np.random.uniform(-1, 1), y_mean, y_std,
                              label=loc, fmt='.')
            ax[i, 1].set_xlabel('Relative humidity')
            ax[i, 1].set_xlim([23, 100])

        ax[i, 0].set_ylabel('Depo')
        ax[i, 0].set_title(f'Months: {month}', weight='bold')
        ax[i, 1].set_title(f'Months: {month}', weight='bold')
        ax[i, 0].legend()
    fig.tight_layout()
    fig.savefig(save_location + f'{ii}_RH_dep_range.png', bbox_inches='tight')
