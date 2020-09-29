import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from matplotlib.colors import LogNorm
import matplotlib
%matplotlib qt

# %%
my_cmap = matplotlib.cm.get_cmap('jet')
my_cmap.set_under('w')
bin_depo = np.linspace(0, 0.5, 50)
bin_month = np.arange(0.5, 13, 1)
bin_time = np.arange(0, 25)

for site in ['46', '54', '33', '53', '34', '32']:
    save_location = 'F:\\halo\\classifier\\summary\\'
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
my_cmap = matplotlib.cm.get_cmap('jet')
my_cmap.set_under('w')

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
bin_depo = np.linspace(0, 0.5, 50)
bin_month = np.arange(0.5, 13, 1)
bin_time = np.arange(0, 25)
df[['location2', 'temp']] = df['location'].str.split('-', expand=True)
pos = {'Uto': 3, 'Hyytiala': 2,
       'Vehmasmaki': 1, 'Sodankyla': 0}
y = {2016: 0, 2017: 1, 2018: 2, 2019: 3}

cbar_max = {'Uto': 600, 'Hyytiala': 600,
            'Vehmasmaki': 400, 'Sodankyla': 400}
fig, axes = plt.subplots(4, 4, figsize=(16, 9), sharex=True, sharey=True)
for key, grp in df.groupby(['year']):
    for k, g in grp.groupby(['location2']):
        H, month_edge, depo_edge = np.histogram2d(
            g['month'], g['depo'],
            bins=(bin_month, bin_depo)
        )
        X, Y = np.meshgrid(month_edge, depo_edge)

        p = axes[pos[k], y[key]].pcolormesh(X, Y, H.T, cmap=my_cmap,
                                            vmin=0.1, vmax=cbar_max[k])
        fig.colorbar(p, ax=axes[pos[k], y[key]])
        axes[pos[k], y[key]].xaxis.set_ticks([4, 8, 12])

for i, val in enumerate(y.keys()):
    axes[0, i].set_title(val, weight='bold', size=15)
    axes[3, i].set_xlabel('Month')
    axes[i, 0].set_ylabel('Depo')

for i, key in enumerate(pos.keys()):
    lab_ax = fig.add_subplot(414 - i, frameon=False)
    lab_ax.tick_params(labelcolor='none', top=False, bottom=False,
                       left=False, right=False)
    lab_ax.set_ylabel(key, weight='bold', labelpad=20, size=15)
fig.tight_layout()
fig.savefig(save_location + 'month_depo.png', bbox_inches='tight')

# %%
fig, axes = plt.subplots(4, 4, figsize=(16, 9), sharex=True, sharey=True)
for key, grp in df.groupby(['year']):
    for k, g in grp.groupby(['location2']):
        dep_mean, time_edge, month_edge, _ = binned_statistic_2d(
            g['time'],
            g['month'],
            g['depo'],
            bins=[bin_time, bin_month],
            statistic=np.nanmean)

        X, Y = np.meshgrid(time_edge, month_edge)

        p = axes[pos[k], y[key]].pcolormesh(X, Y, dep_mean.T, cmap='jet',
                                            vmin=1e-5, vmax=0.3)
        cbar = fig.colorbar(p, ax=axes[pos[k], y[key]])
        if y[key] == 3:
            cbar.ax.set_ylabel('Depolarization ratio')
        axes[pos[k], y[key]].yaxis.set_ticks([4, 8, 12])

for i, val in enumerate(y.keys()):
    axes[0, i].set_title(val, weight='bold', size=15)
    axes[3, i].set_xlabel('Time (hour)')
    axes[i, 0].set_ylabel('Month')

for i, key in enumerate(pos.keys()):
    lab_ax = fig.add_subplot(414 - i, frameon=False)
    lab_ax.tick_params(labelcolor='none', top=False, bottom=False,
                       left=False, right=False)
    lab_ax.set_ylabel(key, weight='bold', labelpad=20, size=15)
fig.tight_layout()
fig.savefig(save_location + 'month_time.png', bbox_inches='tight')

# %%
cbar_max2 = {'Uto': 300, 'Hyytiala': 400,
             'Vehmasmaki': 300, 'Sodankyla': 200}
fig, axes = plt.subplots(4, 4, figsize=(16, 9), sharex=True, sharey=True)
for key, grp in df.groupby(['year']):
    for k, g in grp.groupby(['location2']):
        dep_mean, time_edge, month_edge, _ = binned_statistic_2d(
            g['time'],
            g['month'],
            g['depo'],
            bins=[bin_time, bin_month],
            statistic='count')

        X, Y = np.meshgrid(time_edge, month_edge)

        p = axes[pos[k], y[key]].pcolormesh(X, Y, dep_mean.T, cmap='jet',
                                            vmin=0.001, vmax=cbar_max2[k])
        cbar = fig.colorbar(p, ax=axes[pos[k], y[key]])
        axes[pos[k], y[key]].yaxis.set_ticks([4, 8, 12])

for i, val in enumerate(y.keys()):
    axes[0, i].set_title(val, weight='bold', size=15)
    axes[3, i].set_xlabel('Time (hour)')
    axes[i, 0].set_ylabel('Month')

for i, key in enumerate(pos.keys()):
    lab_ax = fig.add_subplot(414 - i, frameon=False)
    lab_ax.tick_params(labelcolor='none', top=False, bottom=False,
                       left=False, right=False)
    lab_ax.set_ylabel(key, weight='bold', labelpad=20, size=15)
fig.tight_layout()
fig.savefig(save_location + 'month_time_count.png', bbox_inches='tight')
