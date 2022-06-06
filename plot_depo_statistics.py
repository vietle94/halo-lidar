from scipy.stats import binned_statistic_2d
import halo_data as hd
import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import pandas as pd
import xarray as xr
from matplotlib.colors import LogNorm
import calendar
import datetime
import string
%matplotlib qt


# %%
path = r'F:\halo\paper\figures/final_fig/'
sites = ['32', '33', '46', '53', '54']
location_site = ['Uto', 'Hyytiala', 'Vehmasmaki', 'Sodankyla']

df_full = pd.DataFrame({})
for site in sites:
    site_path = r'F:\halo\paper\figures\background_correction_all/stan/' + site + '/'
    file_paths = glob.glob(site_path + '**/*.csv')
    df = pd.concat([pd.read_csv(x) for x in file_paths], ignore_index=True)
    df['site'] = int(site)
    df['time'] = pd.to_datetime(df['time'])
    df_full = df_full.append(df, ignore_index=True)

df_full['location'] = 'Uto'
df_full.loc[df_full['site'] == 33, 'location'] = 'Hyytiala'
df_full.loc[df_full['site'] == 46, 'location'] = 'Hyytiala'
df_full.loc[df_full['site'] == 53, 'location'] = 'Vehmasmaki'
df_full.loc[df_full['site'] == 54, 'location'] = 'Sodankyla'

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
weather = weather.set_index('datetime').resample('1H').mean()
weather = weather.reset_index()

df_full['datetime'] = df_full['time']
# df_full = df_full.drop(['time'], axis=1)
df = pd.merge(weather, df_full)

df = df[(df['depo_corrected'] < 0.5) & (df['depo_corrected'] > -0.1)]
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month

# %%
missing_df = pd.DataFrame({})
for site in ['46', '54', '33', '53', '32']:
    path_ = 'F:/halo/classifier_new/' + site + '/'
    list_files = glob.glob(path_ + '/*.nc', recursive=True)
    time_df = pd.DataFrame(
        {'date': [file.split('\\')[-1][:10] for
                  file in list_files if 'result' not in file],
         'location': [file.split('\\')[-1].split('-')[3] for
                      file in list_files if 'result' not in file]})
    time_df['date'] = pd.to_datetime(time_df['date'])
    time_df['year'] = time_df['date'].dt.year
    time_df['month'] = time_df['date'].dt.month
    time_df_count = time_df.groupby(['year', 'month', 'location']).count()
    time_df_count = time_df_count.reset_index().rename(columns={'date': 'count'})
    missing_df = missing_df.append(time_df_count, ignore_index=True)
missing_df.loc[missing_df['location'] == 'Kuopio', 'location'] = 'Hyytiala'
missing_df = missing_df.set_index(['year', 'month', 'location'])
mux = pd.MultiIndex.from_product([missing_df.index.levels[0],
                                  missing_df.index.levels[1],
                                  missing_df.index.levels[2]],
                                 names=['year', 'month', 'location'])

missing_df = missing_df.reindex(mux, fill_value=0).reset_index()

# %%
df_miss = df.merge(missing_df, 'outer', on=['location', 'year', 'month'])
df_miss = df_miss[~pd.isnull(df_miss.datetime)]
df_miss.reset_index(drop=True, inplace=True)

# %%

df[(df['location'] == 'Uto') &
    (df['datetime'] >= '2019-12-06') &
    (df['datetime'] <= '2019-12-10')] = np.nan

#################################
# %% RH and range
#################################

month_ticklabels = [datetime.date(1900, item, 1).strftime('%b') for item
                    in np.arange(1, 13, 3)]

jitter = {'Uto': [-60, -2], 'Hyytiala': [-30, -1],
          'Vehmasmaki': [0, 0], 'Sodankyla': [30, 1]}
period_months = np.array([[12,  1, 2],
                          [3,  4,  5],
                          [6,  7,  8],
                          [9, 10, 11]])

month_labs = ['Dec-Jan-Feb',
              'Mar-Apr-May',
              'Jun-Jul-Aug',
              'Sep-Oct-Nov']

fig, axes = plt.subplots(2, 2, figsize=(6, 5), sharey=True, sharex=True)
fig2, axes2 = plt.subplots(2, 2, figsize=(6, 5), sharey=True, sharex=True)

for (i, month), ax, ax2 in zip(enumerate(period_months),
                               axes.flatten(), axes2.flatten()):
    # for loc, grp in df[(df.datetime.dt.month >= month[0]) & (df.datetime.dt.month <= month[1])].groupby('location2'):
    for loc, grp in df[df.time.dt.month.isin(month)].groupby('location'):
        grp_avg = grp.groupby('range')['depo_corrected'].mean()
        # grp_count = grp.groupby('range')['depo_corrected'].count()
        grp_std = grp.groupby('range')['depo_corrected'].std()
        # grp_avg = grp_avg[grp_count > 0.01 * sum(grp_count)]
        # grp_std = grp_std[grp_count > 0.01 * sum(grp_count)]
        ax.errorbar(grp_avg.index + jitter[loc][0],
                    grp_avg, grp_std, label=loc,
                    fmt='.', elinewidth=1)
        ax.set_xlim([-50, 2500])

        grp_ground = grp[grp['range'] <= 300]
        y_grp = grp_ground.groupby(pd.cut(grp_ground['RH'], np.arange(0, 105, 5)))
        y_mean = y_grp['depo_corrected'].mean()
        y_std = y_grp['depo_corrected'].std()
        x = np.arange(5, 104, 5)
        ax2.errorbar(x + jitter[loc][1], y_mean, y_std,
                     label=loc, fmt='.', elinewidth=1)
        ax2.set_xlim([20, 105])
    if i in [2, 3]:
        ax.set_xlabel('Range [km, a.g.l]')
        ax.xaxis.set_major_formatter(hd.m_km_ticks())
        ax2.set_xlabel('Relative humidity')
    if i in [0, 2]:
        ax.set_ylabel('Depolarization ratio')
        ax2.set_ylabel('Depolarization ratio')
    ax.set_title(month_labs[i], weight='bold')
    ax2.set_title(month_labs[i], weight='bold')
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4)
handles, labels = ax2.get_legend_handles_labels()
fig2.legend(handles, labels, loc='upper center', ncol=4)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig2.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(path + '/depo_range.png', bbox_inches='tight')
fig2.savefig(path + '/depo_RH.png', bbox_inches='tight')


##################################################
# %% Depo original vs corrected
##################################################
# ticklabels = [datetime.date(1900, item, 1).strftime('%b') for item
#               in np.arange(1, 13, 3)]
jitter = {}
for place, v in zip(['Uto', 'Hyytiala', 'Vehmasmaki', 'Sodankyla'],
                    np.linspace(-0.15, 0.15, 4)):
    jitter[place] = v

fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
# for k, grp in df.groupby(['location']):
group = df.groupby(['location'])
for k in location_site:
    grp = group.get_group(k)
    grp_ = grp.groupby(grp.datetime.dt.month)['depo_corrected']
    median_plot = grp_.median()
    median25_plot = grp_.agg(lambda x: np.nanpercentile(x, q=25))
    median75_plot = grp_.agg(lambda x: np.nanpercentile(x, q=75))

    ax[0].errorbar(median_plot.index + jitter[k],
                   median_plot,
                   yerr=(median_plot - median25_plot,
                         median75_plot - median_plot),
                   label=k, marker='o',
                   fmt='--', elinewidth=1)

    ax[0].set_xticks(np.arange(1, 13, 3))
    ax[0].set_xticklabels(['Jan', 'April', 'July', 'Oct'])
    ax[0].set_ylabel('$\delta$')
    ax[0].grid(axis='x', which='major', linewidth=0.5, c='silver')
    ax[0].grid(axis='y', which='major', linewidth=0.5, c='silver')
    ax[0].legend()
    ax[0].set_title('Depo corrected')

# for k, grp in df.groupby(['location']):
group = df.groupby(['location'])
for k in location_site:
    grp = group.get_group(k)
    grp_ = grp.groupby(grp.datetime.dt.month)['depo']
    median_plot = grp_.median()
    median25_plot = grp_.agg(lambda x: np.nanpercentile(x, q=25))
    median75_plot = grp_.agg(lambda x: np.nanpercentile(x, q=75))

    ax[1].errorbar(median_plot.index + jitter[k],
                   median_plot,
                   yerr=(median_plot - median25_plot,
                         median75_plot - median_plot),
                   label=k, marker='o',
                   fmt='--', elinewidth=1)

    ax[1].set_xticks(np.arange(1, 13, 3))
    ax[1].set_xticklabels(['Jan', 'April', 'July', 'Oct'])
    ax[1].grid(axis='x', which='major', linewidth=0.5, c='silver')
    ax[1].grid(axis='y', which='major', linewidth=0.5, c='silver')
    ax[1].legend()
    ax[1].set_title('Depo original')
fig.savefig(path + '/depo_original_corrected.png', bbox_inches='tight')


fig, ax = plt.subplots(1, figsize=(9, 6))
# group = df.groupby(['location'])
# for k in location_site:
#     grp = group.get_group(k)
group = df_miss[df_miss['count'] > 15].groupby(['location'])
for k in location_site:
    grp = group.get_group(k)
    grp_ = grp.groupby(grp.datetime.dt.month)['depo_corrected']
    median_plot = grp_.median()
    median25_plot = grp_.agg(lambda x: np.nanpercentile(x, q=25))
    median75_plot = grp_.agg(lambda x: np.nanpercentile(x, q=75))

    ax.errorbar(median_plot.index + jitter[k],
                median_plot,
                yerr=(median_plot - median25_plot,
                      median75_plot - median_plot),
                label=k, marker='o',
                fmt='--', elinewidth=1)

    ax.set_xticks(np.arange(1, 13, 3))
    ax.set_xticklabels(['Jan', 'April', 'July', 'Oct'])
    ax.set_ylabel('$\delta$')
    ax.grid(axis='x', which='major', linewidth=0.5, c='silver')
    ax.grid(axis='y', which='major', linewidth=0.5, c='silver')
    ax.set_xlabel('Month')
    ax.legend()
fig.savefig(path + '/monthly_median.png', bbox_inches='tight')

#########################################################
# %% Monthly hour
#########################################################
bin_month = np.arange(0.5, 13, 1)
bin_time = np.arange(0, 25)
X, Y = np.meshgrid(bin_time, bin_month)
fig, axes = plt.subplots(2, 2, figsize=(6, 4), sharex=True, sharey=True)

group = df_miss[df_miss['count'] > 15].groupby(['location'])
for k, ax in zip(location_site, axes.flatten()):
    grp = group.get_group(k)
# for (k, grp), ax in zip(df_miss[df_miss['count'] > 15].groupby(['location']), axes.flatten()):
    print(k)
    dep_mean, time_edge, month_edge, _ = binned_statistic_2d(
        grp.datetime.dt.hour,
        grp.datetime.dt.month,
        grp['depo'],
        bins=[bin_time, bin_month],
        statistic=np.nanmedian)
    p = ax.pcolormesh(X, Y, dep_mean.T,
                      cmap='jet',
                      vmin=1e-5, vmax=0.3)
    cbar = fig.colorbar(p, ax=ax)
    cbar.ax.set_ylabel('$\delta$')
    ax.set_xticks(np.arange(0, 24, 6))
    ax.set_yticks(np.arange(1, 13, 3))

for ax in [axes[0, 0], axes[1, 0]]:
    ax.set_yticklabels(month_ticklabels)
for ax in [axes[1, 0], axes[1, 1]]:
    ax.set_xlabel('Hour')
for n, ax in enumerate(axes.flatten()):
    ax.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
            transform=ax.transAxes, size=12)

fig.tight_layout()
fig.savefig(path + '/' + 'month_time_median.png',
            bbox_inches='tight')

#########################################################
# %% Monthly range
#########################################################
bin_month = np.arange(0.5, 13, 1)
bin_range = np.arange(105, 3020, 30)
X, Y = np.meshgrid(bin_month, bin_range)
fig, axes = plt.subplots(2, 2, figsize=(6, 4), sharex=True, sharey=True)

group = df_miss[df_miss['count'] > 15].groupby(['location'])
for k, ax in zip(location_site, axes.flatten()):
    grp = group.get_group(k)
# for (k, grp), ax in zip(df_miss[df_miss['count'] > 15].groupby(['location']), axes.flatten()):
    print(k)
    dep_mean, month_edge, range_edge, _ = binned_statistic_2d(
        grp.datetime.dt.month,
        grp['range'],
        grp['depo'],
        bins=[bin_month, bin_range],
        statistic=np.nanmean)
    # dep_count, month_edge, range_edge, _ = binned_statistic_2d(
    #     grp.datetime.dt.month,
    #     grp['range'],
    #     grp['depo'],
    #     bins=[bin_month, bin_range],
    #     statistic=np.nanmean)
    # dep_count = dep_count/np.nansum(dep_count, axis=1)
    # (dep_count.T/np.nansum(dep_count.T, axis=0)).shape
    p = ax.pcolormesh(X, Y, dep_mean.T,
                      cmap='jet',
                      vmin=1e-5, vmax=0.3)
    cbar = fig.colorbar(p, ax=ax)
    cbar.ax.set_ylabel('$\delta$')
    ax.set_xticks(np.arange(1, 13, 3))
    ax.set_yticks(np.arange(4)*1000)
    ax.set_ylim(bottom=0)

for ax in [axes[0, 0], axes[1, 0]]:
    ax.set_ylabel('Height a.g.l [km]')
    ax.yaxis.set_major_formatter(hd.m_km_ticks())
for ax in [axes[1, 0], axes[1, 1]]:
    ax.set_xticklabels(month_ticklabels)
for n, ax in enumerate(axes.flatten()):
    ax.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
            transform=ax.transAxes, size=12)

fig.tight_layout()
fig.savefig(path + '/' + 'month_range.png',
            bbox_inches='tight')

###################################################
# %% Depo month
###################################################

cbar_max = {'Uto': 600*3, 'Hyytiala': 600*3,
            'Vehmasmaki': 400*3, 'Sodankyla': 400*3}
# cbar_max = {'Uto': 600, 'Hyytiala': 450,
#             'Vehmasmaki': 330, 'Sodankyla': 260}
bin_depo = np.linspace(0, 0.5, 50)
X, Y = np.meshgrid(bin_month, bin_depo)
group = df_miss[df_miss['count'] > 15].groupby(['location'])
for k, ax in zip(location_site, axes.flatten()):
    grp = group.get_group(k)
# for k, grp in df_miss[df_miss['count'] > 15].groupby(['location']):
    print(k)
    fig, axes = plt.subplots(2, 2, figsize=(6, 4), sharex=True)
    axes = axes.flatten()
    if k == 'Sodankyla':
        fig.delaxes(axes[0])
        axes = axes[1:]
    for (key, g), ax in zip(grp.groupby(['year']), axes):
        H, month_edge, depo_edge = np.histogram2d(
            g['month'], g['depo'],
            bins=(bin_month, bin_depo)
        )
        # miss = missing_df[(missing_df['year'] == key) &
        #                   (missing_df['location2'] == k)]['month']
        # if len(miss.index) != 0:
        #     for miss_month in miss:
        #         H[miss_month-1, :] = np.nan
        p = ax.pcolormesh(X, Y, H.T, cmap='jet',
                          vmin=0.1, vmax=cbar_max[k])
        fig.colorbar(p, ax=ax)
        ax.xaxis.set_ticks([4, 8, 12])
        ax.set_ylabel('Depolarization ratio')
        ax.set_title(int(key), weight='bold')
        fig.tight_layout()

###############################################
# %%
###############################################
bin_depo = np.linspace(0, 0.5, 50)
X, Y = np.meshgrid(bin_month, bin_depo)
month_ticklabels = [datetime.date(1900, item, 1).strftime('%b') for item
                    in np.arange(1, 13)]
group = df_miss[df_miss['count'] > 15].groupby(['location'])
for k, ax in zip(location_site, axes.flatten()):
    grp = group.get_group(k)
# for k, grp in df_miss[df_miss['count'] > 15].groupby(['location']):
    fig = plt.figure(figsize=(16, 9))
    subfigs = fig.subfigures(2, 2, wspace=0.1, hspace=0.1)

    gs = subfigs[0, 0].add_gridspec(nrows=4, ncols=15, wspace=0.5)
    ax0 = subfigs[0, 0].add_subplot(gs[:3, :-1])
    ax0_cbar = subfigs[0, 0].add_subplot(gs[:3, -1])
    ax0_ = subfigs[0, 0].add_subplot(gs[-1, :-1], sharex=ax0)

    gs = subfigs[0, 1].add_gridspec(nrows=4, ncols=15, wspace=0.5)
    ax1 = subfigs[0, 1].add_subplot(gs[:3, :-1])
    ax1_cbar = subfigs[0, 1].add_subplot(gs[:3, -1])
    ax1_ = subfigs[0, 1].add_subplot(gs[-1, :-1], sharex=ax1)

    gs = subfigs[1, 0].add_gridspec(nrows=4, ncols=15, wspace=0.5)
    ax2 = subfigs[1, 0].add_subplot(gs[:3, :-1])
    ax2_cbar = subfigs[1, 0].add_subplot(gs[:3, -1])
    ax2_ = subfigs[1, 0].add_subplot(gs[-1, :-1], sharex=ax2)

    if k != 'Sodankyla':

        gs = subfigs[1, 1].add_gridspec(nrows=4, ncols=15, wspace=0.5)
        ax3 = subfigs[1, 1].add_subplot(gs[:3, :-1])
        ax3_cbar = subfigs[1, 1].add_subplot(gs[:3, -1])
        ax3_ = subfigs[1, 1].add_subplot(gs[-1, :-1], sharex=ax3)

    for (key, g), ax, ax_, ax_cbar in zip(grp.groupby(['year']), [ax0, ax1, ax2, ax3],
                                          [ax0_, ax1_, ax2_, ax3_],
                                          [ax0_cbar, ax1_cbar, ax2_cbar, ax3_cbar]):
        H, month_edge, depo_edge = np.histogram2d(
            g['month'], g['depo'],
            bins=(bin_month, bin_depo))
        H_sum = np.sum(H, axis=1)
        H_plot = H.T/H_sum.T
        H_plot[H_plot == 0] = np.nan
        p = ax.pcolormesh(X, Y, H_plot, cmap='jet',
                          vmin=0, vmax=0.1)
        cbar = fig.colorbar(p, ax=ax, cax=ax_cbar)
        cbar.ax.set_ylabel('Monthly fraction')
        ax.set_xticklabels([])
        ax.set_ylabel('$\\delta$')

        ax_.bar(np.arange(1, 13), H_sum)
        ax_.set_xticks(np.arange(1, 13))

    for n, ax in enumerate([ax0, ax1, ax2, ax3]):
        ax.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
                transform=ax.transAxes, size=12)

    for ax in [ax0_, ax1_, ax2_, ax3_]:
        ax.set_xticklabels(month_ticklabels)
        ax.set_ylabel('N')

    for ax in [ax0, ax1, ax2, ax3]:
        plt.setp(ax.get_xticklabels(), visible=False)

    fig.savefig(path + k + '.png',
                bbox_inches='tight')

###############################################
# %%
###############################################
