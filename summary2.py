import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from matplotlib.colors import LogNorm
import matplotlib
import copy
%matplotlib qt

# %%
my_cmap = copy.copy(matplotlib.cm.get_cmap('jet'))
my_cmap.set_under('w')
bin_depo = np.linspace(0, 0.5, 50)
bin_month = np.arange(0.5, 13, 1)
bin_time = np.arange(0, 25)
save_location = 'F:\\halo\\classifier\\summary\\'

for site in ['46', '54', '33', '53', '34', '32']:
    df = pd.read_csv('F:\\halo\\classifier\\' + site + '\\result.csv')

    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['depo'][df['depo'] > 1] = np.nan

    for key, grp in df.groupby('year'):
        title = str(key) + '_' + df['location'][0]
        H, month_edge, depo_edge = np.histogram2d(
            grp['month'], grp['depo'],
            bins=(bin_month, bin_depo)
        )
        X, Y = np.meshgrid(month_edge, depo_edge)

        fig, ax = plt.subplots(figsize=(12, 6))
        p = ax.pcolormesh(X, Y, H.T, cmap=my_cmap, vmin=0.001)
        fig.colorbar(p, ax=ax)
        ax.set_xlabel('Month')
        ax.set_ylabel('Depolarization ratio')
        ax.set_title(title, size=22, weight='bold')
        fig.savefig(save_location + title + '_month_depo.png', bbox_inches='tight')

        dep_mean, time_edge, month_edge, _ = binned_statistic_2d(
            grp['time'],
            grp['month'],
            grp['depo'],
            bins=[bin_time, bin_month],
            statistic=np.nanmean)

        X, Y = np.meshgrid(time_edge, month_edge)

        fig, ax = plt.subplots(figsize=(12, 6))
        p = ax.pcolormesh(X, Y, dep_mean.T, cmap=my_cmap, vmin=0.001, vmax=0.3)
        ax.set_xlabel('Time')
        ax.set_ylabel('Month')
        cbar = fig.colorbar(p, ax=ax)
        cbar.ax.set_ylabel('Depolarization ratio')
        cbar.ax.yaxis.set_label_position('left')
        ax.set_title(title, size=22, weight='bold')
        fig.savefig(save_location + title + '_month_hour.png', bbox_inches='tight')
        plt.close('all')

# %%
df = pd.DataFrame()
for site in ['46', '54', '33', '53', '32']:
    save_location = 'F:\\halo\\classifier\\summary\\'
    df = df.append(pd.read_csv('F:\\halo\\classifier\\' + site + '\\result.csv'),
                   ignore_index=True)

df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['depo'][df['depo'] > 1] = np.nan
df.loc[df['location'] == 'Kuopio-33', 'location'] = 'Hyytiala-33'

# %%
df[['location2', 'temp']] = df['location'].str.split('-', expand=True)
pos = {'Uto': 3, 'Hyytiala': 2,
       'Vehmasmaki': 1, 'Sodankyla': 0}
y = {2016: 0, 2017: 1, 2018: 2, 2019: 3}
cbar_max = {'Uto': 600, 'Hyytiala': 600,
            'Vehmasmaki': 400, 'Sodankyla': 400}
avg = {}

X, Y = np.meshgrid(month_edge, depo_edge)

fig, axes = plt.subplots(4, 5, figsize=(18, 9), sharex=True, sharey=True)
for key, grp in df.groupby(['year']):
    for k, g in grp.groupby(['location2']):
        H, month_edge, depo_edge = np.histogram2d(
            g['month'], g['depo'],
            bins=(bin_month, bin_depo)
        )

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
X, Y = np.meshgrid(time_edge, month_edge)
fig, axes = plt.subplots(4, 5, figsize=(18, 9), sharex=True, sharey=True)
for key, grp in df.groupby(['year']):
    for k, g in grp.groupby(['location2']):
        dep_mean, time_edge, month_edge, _ = binned_statistic_2d(
            g['time'],
            g['month'],
            g['depo'],
            bins=[bin_time, bin_month],
            statistic=np.nanmean)

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

X, Y = np.meshgrid(time_edge, month_edge)
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
df = pd.read_csv('smeardata_20160101120000.csv')
df['time'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']])
df = df.drop(['HYY_META.SO21250'], axis=1)
df = df.rename(columns={'HYY_META.CO1250': 'CO',
                        'HYY_META.O31250': 'O3'})

# %%
fig, axes = plt.subplots(2, 4, figsize=(16, 5),
                         sharex=True, sharey=True)
for year, grp_year in df.groupby('Year'):
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
