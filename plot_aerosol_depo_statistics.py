from decimal import Decimal
import statsmodels.api as sm
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
from sklearn.metrics import r2_score
import matplotlib.ticker as mtick
from matplotlib.ticker import FuncFormatter
%matplotlib qt


# %%
path = r'F:\halo\paper\figures\depo_aerosol/'
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

df[(df['location'] == 'Uto') &
    (df['datetime'] >= '2019-12-06') &
    (df['datetime'] <= '2019-12-10')] = np.nan

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
bad_site = df_miss['site'].astype('int') == 33
bad_range = (df_miss['range'] > 400) & (df_miss['range'] < 470)
df_miss.loc[bad_site & bad_range, 'depo_corrected'] = np.nan

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

fig2, axes2 = plt.subplots(2, 2, figsize=(7, 4), sharey=True, sharex=True)

for (i, month), ax2 in zip(enumerate(period_months), axes2.flatten()):
    # group = df[df.time.dt.month.isin(month)].groupby('location')
    group = df_miss[df_miss.time.dt.month.isin(month) & (
        df_miss['depo_corrected_sd'] < 0.05)].groupby(['location'])
    for loc in location_site:
        grp = group.get_group(loc)
        grp_ground = grp[grp['range'] <= 300]
        y_grp = grp_ground.groupby(pd.cut(grp_ground['RH'], np.arange(0, 105, 5)))
        y_count = y_grp['depo_corrected'].count()
        y_mean = y_grp['depo_corrected'].mean()
        y_std = y_grp['depo_corrected'].std()
        y_mean = y_mean[y_count > 0.01 * sum(y_count)]
        y_std = y_std[y_count > 0.01 * sum(y_count)]
        x = np.arange(5, 104, 5) - 2.5
        ax2.errorbar(x[y_count > 0.01 * sum(y_count)] + jitter[loc][1], y_mean, y_std,
                     label=loc, fmt='.', elinewidth=1)
        ax2.set_xlim([20, 105])
        ax2.grid(visible=True, which='major', axis='both')
        ax2.xaxis.set_major_formatter(mtick.PercentFormatter(100))
        ax2.set_yticks([0, 0.1, 0.2])
    if i in [2, 3]:
        ax2.set_xlabel('RH')

    if i in [0, 2]:
        ax2.set_ylabel('$\delta$')
    # ax.set_title(month_labs[i], weight='bold')
    # ax2.set_title(month_labs[i], weight='bold')
handles, labels = ax2.get_legend_handles_labels()
fig2.legend(handles, labels, loc='lower center', ncol=4)
# fig.tight_layout(rect=(0, 0.05, 1, 0.9))
# fig2.tight_layout(rect=(0, 0.05, 1, 0.9))
fig2.subplots_adjust(bottom=0.2, hspace=0.3)


for n, ax in enumerate(axes2.flatten()):
    ax.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
            transform=ax.transAxes, size=12)

fig2.savefig(path + '/depo_RH.png', bbox_inches='tight')

# %%


def regress(data, yvar, xvars):
    Y = data[yvar]
    X = data[xvars]
    X['intercept'] = 1.
    result = sm.OLS(Y, X).fit()
    return [np.around(i, digit) for i, digit in zip([result.params[0], result.pvalues[0], result.rsquared],
                                                    [3, 3, 3])]


temp = df_miss[df_miss['depo_corrected_sd'] < 0.05].copy()
temp['RH'] = temp['RH']/100
temp = temp.dropna(axis=0)

for (i, month) in enumerate(period_months):
    print(month,
          temp[temp.datetime.dt.month.isin(month)].groupby('location').apply(regress, 'depo_corrected', ['RH']))


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

##################################################
# %% Depo median month
##################################################
jitter = {}
for place, v in zip(['Uto', 'Hyytiala', 'Vehmasmaki', 'Sodankyla'],
                    np.linspace(-0.15, 0.15, 4)):
    jitter[place] = v
fig, ax = plt.subplots(1, figsize=(6, 4))
# group = df.groupby(['location'])
# for k in location_site:
#     grp = group.get_group(k)
# group = df_miss[df_miss['count'] > 15].groupby(['location'])
group = df_miss[(df_miss['count'] > 15) & (
    df_miss['depo_corrected_sd'] < 0.05)].groupby(['location'])
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
    # ax.set_xlabel('Month')
    ax.legend()
fig.savefig(path + '/monthly_median.png', bbox_inches='tight')

##################################################
# %% Overall depo
##################################################
group['depo_corrected'].median()
group['depo_corrected'].agg(lambda x: np.nanpercentile(x, q=25))
group['depo_corrected'].agg(lambda x: np.nanpercentile(x, q=75))

# %%
fig, ax = plt.subplots(1, 4, figsize=(9, 2), sharex=True, sharey=True)
for k, ax_ in zip(location_site, ax):
    grp = group.get_group(k)
    weights = np.ones_like(grp['depo_corrected'])/float(len(grp['depo_corrected']))
    ax_.hist(grp['depo_corrected'], bins=np.linspace(-0.1, 0.4, 11),
             weights=weights)
    ax_.set_xlabel('$\delta$')
ax[0].set_ylabel('Relative frequency')

for n, ax_ in enumerate(ax.flatten()):
    ax_.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
             transform=ax_.transAxes, size=12)
    ax_.grid()
fig.subplots_adjust(hspace=0.5)
fig.savefig(path + '/site_hist.png', bbox_inches='tight')

#########################################################
# %% Monthly hour
#########################################################
bin_month = np.arange(0.5, 13, 1)
bin_time = np.arange(0, 25)
X, Y = np.meshgrid(bin_time, bin_month)
fig, axes = plt.subplots(2, 2, figsize=(6, 4), sharex=True, sharey=True)

# group = df_miss[df_miss['count'] > 15].groupby(['location'])
group = df_miss[(df_miss['count'] > 15) & (
    df_miss['depo_corrected_sd'] < 0.05)].groupby(['location'])
for k, ax in zip(location_site, axes.flatten()):
    grp = group.get_group(k)
# for (k, grp), ax in zip(df_miss[df_miss['count'] > 15].groupby(['location']), axes.flatten()):
    print(k)
    dep_mean, time_edge, month_edge, _ = binned_statistic_2d(
        grp.datetime.dt.hour,
        grp.datetime.dt.month,
        grp['depo_corrected'],
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

# group = df_miss[df_miss['count'] > 15].groupby(['location'])
group = df_miss[(df_miss['count'] > 15) & (
    df_miss['depo_corrected_sd'] < 0.05)].groupby(['location'])
for k, ax in zip(location_site, axes.flatten()):
    grp = group.get_group(k)
    print(k)
    dep_mean, month_edge, range_edge, _ = binned_statistic_2d(
        grp.datetime.dt.month,
        grp['range'],
        grp['depo_corrected'],
        bins=[bin_month, bin_range],
        statistic=np.nanmean)
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
            g['month'], g['depo_corrected'],
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
bin_range = np.arange(105, 3020, 90)
bin_month = np.arange(0.5, 13, 1)
bin_time = np.arange(0, 25)
bin_depo = np.linspace(0, 0.3, 30)
X, Y = np.meshgrid(bin_month, bin_depo)
month_ticklabels = [datetime.date(1900, item, 1).strftime('%b') for item
                    in np.arange(1, 13, 3)]
# group = df_miss[df_miss['count'] > 15].groupby(['location'])
group = df_miss[(df_miss['count'] > 15) & (
    df_miss['depo_corrected_sd'] < 0.05)].groupby(['location'])
for k in location_site:
    grp = group.get_group(k)
    print(k)
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(nrows=12, ncols=60, wspace=0, hspace=2, left=0.05, right=0.9)
    gs_bar = fig.add_gridspec(nrows=12, ncols=1, wspace=0, hspace=2, left=0.92, right=0.94)
    if k != 'Sodankyla':
        ax0 = fig.add_subplot(gs[6:9, :15])
        ax0_range = fig.add_subplot(gs[:3, :15], sharex=ax0)
        ax0_range_count = fig.add_subplot(gs[3:6, :15], sharex=ax0)
        ax0_hour = fig.add_subplot(gs[9:12, :15], sharex=ax0)

    ax1 = fig.add_subplot(gs[6:9, 15:30], sharey=ax0)
    ax1_range = fig.add_subplot(gs[:3, 15:30], sharex=ax1, sharey=ax0_range)
    ax1_range_count = fig.add_subplot(gs[3:6, 15:30], sharex=ax1, sharey=ax0_range_count)
    ax1_hour = fig.add_subplot(gs[9:12, 15:30], sharex=ax1, sharey=ax0_hour)

    ax2 = fig.add_subplot(gs[6:9, 30:45], sharey=ax0)
    ax2_range = fig.add_subplot(gs[:3, 30:45], sharex=ax2, sharey=ax0_range)
    ax2_range_count = fig.add_subplot(gs[3:6, 30:45], sharex=ax2, sharey=ax0_range_count)
    ax2_hour = fig.add_subplot(gs[9:12, 30:45], sharex=ax2, sharey=ax0_hour)

    ax3 = fig.add_subplot(gs[6:9, 45:60], sharey=ax0)
    ax3_range = fig.add_subplot(gs[:3, 45:60], sharex=ax3, sharey=ax0_range)
    ax3_range_count = fig.add_subplot(gs[3:6, 45:60], sharex=ax3, sharey=ax0_range_count)
    ax3_hour = fig.add_subplot(gs[9:12, 45:60], sharex=ax3, sharey=ax0_hour)

    ax_cbar = fig.add_subplot(gs_bar[6:9])
    ax_range_cbar = fig.add_subplot(gs_bar[:3])
    ax_range_count_cbar = fig.add_subplot(gs_bar[3:6])
    ax_hour_cbar = fig.add_subplot(gs_bar[9:12])
    label_holder = 0
    year_ax = {2016: [ax0, ax0_range, ax0_range_count, ax0_hour],
               2017: [ax1, ax1_range, ax1_range_count, ax1_hour],
               2018: [ax2, ax2_range, ax2_range_count, ax2_hour],
               2019: [ax3, ax3_range, ax3_range_count, ax3_hour]}
    grp_year = grp.groupby('year')
    for key in grp.year.unique():
        g = grp_year.get_group(key)
        ax, ax_range, ax_range_count, ax_hour = year_ax[key]

        label_holder += 1

        # Month depo
        H, month_edge, depo_edge = np.histogram2d(
            g['month'], g['depo_corrected'],
            bins=(bin_month, bin_depo))
        H_sum = np.sum(H, axis=1)
        H_plot = H.T/H_sum.T
        H_plot[H_plot == 0] = np.nan
        p = ax.pcolormesh(X, Y, H_plot*100, cmap='viridis',
                          vmin=0, vmax=10)
        cbar = fig.colorbar(p, ax=ax, cax=ax_cbar)
        cbar.ax.set_ylabel('%N')
        ax.set_xticklabels([])

        # Month total count
        # ax_.bar(np.arange(1, 13), H_sum)
        # ax_.set_xticks(np.arange(1, 13, 3))

        # Month range median
        dep_mean, month_edge, range_edge, _ = binned_statistic_2d(
            g.datetime.dt.month,
            g['range'],
            g['depo_corrected'],
            bins=[bin_month, bin_range],
            statistic=np.nanmedian)
        # Month range count
        dep_count, month_edge, range_edge = np.histogram2d(
            g.datetime.dt.month,
            g['range'],
            bins=[bin_month, bin_range])
        dep_count[dep_count == 0] = np.nan
        dep_mean[dep_count < 30] = np.nan

        p = ax_range.pcolormesh(month_edge, range_edge, dep_mean.T, cmap='jet',
                                vmin=0, vmax=0.3)
        cbar = fig.colorbar(p, ax=ax_range, cax=ax_range_cbar)
        cbar.ax.set_ylabel('$\\delta$')
        ax_range.set_xticklabels([])
        ax_range.yaxis.set_major_formatter(hd.m_km_ticks())

        p = ax_range_count.pcolormesh(month_edge, range_edge, dep_count.T, cmap='viridis',
                                      vmin=0, vmax=2000)
        cbar = fig.colorbar(p, ax=ax_range_count, cax=ax_range_count_cbar)
        cbar.ax.set_ylabel('$N$')
        ax_range_count.set_xticklabels([])
        ax_range_count.yaxis.set_major_formatter(hd.m_km_ticks())

        # Month hour median
        dep_mean, month_edge, time_edge, _ = binned_statistic_2d(
            g.datetime.dt.month,
            g.datetime.dt.hour,
            g['depo_corrected'],
            bins=[bin_month, bin_time],
            statistic=np.nanmedian)

        p = ax_hour.pcolormesh(month_edge, time_edge, dep_mean.T,
                               cmap='jet',
                               vmin=1e-5, vmax=0.3)
        cbar = fig.colorbar(p, ax=ax_hour, cax=ax_hour_cbar)
        cbar.ax.set_ylabel('$\delta$')

        ax_hour.set_xticks(np.arange(1, 13, 3))

        if label_holder < 2:
            ax.set_ylabel('$\\delta$')
            ax_range.set_ylabel('Height a.g.l [km]')
            ax_range_count.set_ylabel('Height a.g.l [km]')
            ax_hour.set_yticks(np.arange(0, 24, 6))
            ax_hour.set_ylabel('Hour')

        if label_holder > 1:
            plt.setp(ax.get_yticklabels(), visible=False)
            plt.setp(ax_range.get_yticklabels(), visible=False)
            plt.setp(ax_range_count.get_yticklabels(), visible=False)
            plt.setp(ax_hour.get_yticklabels(), visible=False)

        ax_hour.text(-0.0, -0.4, int(key), weight='bold',
                     transform=ax_hour.transAxes, size=12)

    for n, ax in enumerate([ax0_range, ax0_range_count, ax0, ax0_hour]):
        ax.text(0, 1.05, '(' + string.ascii_lowercase[n] + ')',
                transform=ax.transAxes, size=12)

    if k == 'Sodankyla':
        for n, ax in enumerate([ax1_range, ax1_range_count, ax1, ax1_hour]):
            ax.text(0, 1.05, '(' + string.ascii_lowercase[n] + ')',
                    transform=ax.transAxes, size=12)

    for ax in [ax0_hour, ax1_hour, ax2_hour, ax3_hour]:
        ax.set_xticklabels(month_ticklabels)

    for ax in [ax0, ax1, ax2, ax3, ax0_range, ax1_range, ax2_range, ax3_range,
               ax0_range_count, ax1_range_count, ax2_range_count, ax3_range_count]:
        plt.setp(ax.get_xticklabels(), visible=False)
        # ax.tick_params(axis='x',         # changes apply to the x-axis
        #                which='both',      # both major and minor ticks are affected
        #                bottom=False,      # ticks along the bottom edge are off
        #                top=False,         # ticks along the top edge are off
        #                labelbottom=False)

    for ax in [ax1, ax2, ax3, ax1_range, ax2_range, ax3_range,
               ax1_range_count, ax2_range_count, ax3_range_count,
               ax1_hour, ax2_hour, ax3_hour]:
        ax.tick_params(axis='y',         # changes apply to the x-axis
                       which='both',      # both major and minor ticks are affected
                       left=False)

    for ax in [ax1, ax2, ax1_range, ax2_range,
               ax1_range_count, ax2_range_count,
               ax1_hour, ax2_hour]:
        ax.spines['right'].set_color('none')
        ax.spines['left'].set_color('none')

    for ax in [ax0, ax0_range, ax0_range_count, ax0_hour]:
        ax.spines['right'].set_color('none')

    for ax in [ax3, ax3_range, ax3_range_count, ax3_hour]:
        ax.spines['left'].set_color('none')

    if k == 'Sodankyla':
        for ax in [ax1, ax1_range, ax1_range_count, ax1_hour]:
            ax.spines['left'].set_color('black')
    fig.savefig(path + '/sites/' + k + '.png', bbox_inches='tight')

###############################
# %%
###############################
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
props = dict(boxstyle='round', facecolor='wheat', alpha=1)
group = df_miss.groupby(['location'])
fig, axes = plt.subplots(4, 4, figsize=(16, 9), sharey='True', sharex=True)
for site, axes_row in zip(['Uto', 'Hyytiala', 'Vehmasmaki', 'Sodankyla'], axes):
    grp = group.get_group(site)
    for period, ax in zip(period_months, axes_row):
        temp = grp.loc[grp['month'].isin(period), ['RH', 'depo_corrected']]
        temp.dropna(axis=0, inplace=True)

        H, RH_edges, depo_corrected_edges = np.histogram2d(
            temp['RH']/100,
            temp['depo_corrected'],
            bins=100)
        z = np.polyfit(temp['RH']/100,
                       temp['depo_corrected'], 1)
        y_hat = np.poly1d(z)(temp['RH']/100)
        mod = sm.OLS(temp['depo_corrected'], sm.add_constant(temp['RH']/100))
        fii = mod.fit()
        p_values = fii.summary2().tables[1]['P>|t|']

        ax.plot(temp['RH']/100,
                y_hat, "r-", lw=1)
        text = f"$y={z[0]:0.3f}\;x{z[1]:+0.3f}$\n$R^2$ = {r2_score(temp['depo_corrected'],y_hat):0.3f}, p={p_values[1]:.3f}"
        ax.text(0.05, 0.95, text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', bbox=props)
        X, Y = np.meshgrid(RH_edges, depo_corrected_edges)
        H[H == 0] = np.nan
        p = ax.pcolormesh(X, Y, H.T)
        cbar = fig.colorbar(p, ax=ax)
        cbar.ax.set_ylabel('N')
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        # ax.plot(temp['RH'],
        #         temp['depo_corrected'], "+",
        #         ms=5, mec="k", alpha=0.005)
        # z = np.polyfit(temp['RH'],
        #                temp['depo_corrected'], 1)
        # y_hat = np.poly1d(z)(temp['RH'])
        #
        # ax.plot(temp['RH'],
        #         y_hat, "r-", lw=1)
        # text = f"$y={z[0]:0.3f}\;x{z[1]:+0.3f}$\n$R^2 = {r2_score(temp['depo_corrected'],y_hat):0.3f}$"
        # ax.text(0.05, 0.95, text, transform=ax.transAxes,
        #         fontsize=10, verticalalignment='top', bbox=props)
    # fig.savefig('F:/halo/paper/figures/RH_depo_point' + site,
    #             dpi=150, bbox_inches='tight')
for n, ax in enumerate(axes.flatten()):
    ax.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
            transform=ax.transAxes, size=12)
for ax in axes_row:
    ax.set_xlabel('RH')
for ax in axes[:, 0]:
    ax.set_ylabel('$\delta$')

fig.savefig(path + 'RH.png', bbox_inches='tight')

# %%
mod = sm.OLS(temp['depo_corrected'], sm.add_constant(temp['RH']))
fii = mod.fit()
fii.summary()
# p_values = fii.summary2().tables[1]['P>|t|']

###############################################
# %%
###############################################
bin_range = np.arange(105, 3020, 90)
bin_month = np.arange(0.5, 13, 1)
bin_time = np.arange(0, 25)
bin_depo = np.linspace(0, 0.3, 30)
X, Y = np.meshgrid(bin_month, bin_depo)
month_ticklabels = [datetime.date(1900, item, 1).strftime('%b') for item
                    in np.arange(1, 13, 3)]
# group = df_miss[df_miss['count'] > 15].groupby(['location'])
group = df_miss[(df_miss['count'] > 15) & (
    df_miss['depo_corrected_sd'] < 0.05)].groupby(['location'])
# fig, axes = plt.subplots(3, 4, figsize=(12, 6), sharex=True, sharey='row')
fig = plt.figure(figsize=(9, 6))
gs = fig.add_gridspec(nrows=9, ncols=60, left=0.05, right=0.9)
gs_bar = fig.add_gridspec(nrows=9, ncols=1, left=0.92, right=0.94)
ax0 = fig.add_subplot(gs[:3, :15])
ax0_range = fig.add_subplot(gs[3:6, :15], sharex=ax0)
ax0_hour = fig.add_subplot(gs[6:9, :15], sharex=ax0)
ax_Uto = [ax0, ax0_range, ax0_hour]

ax1 = fig.add_subplot(gs[:3, 15:30], sharey=ax0)
ax1_range = fig.add_subplot(gs[3:6, 15:30], sharex=ax1, sharey=ax0_range)
ax1_hour = fig.add_subplot(gs[6:9, 15:30], sharex=ax1, sharey=ax0_hour)
ax_Hyytiala = [ax1, ax1_range, ax1_hour]

ax2 = fig.add_subplot(gs[:3, 30:45], sharey=ax0)
ax2_range = fig.add_subplot(gs[3:6, 30:45], sharex=ax2, sharey=ax0_range)
ax2_hour = fig.add_subplot(gs[6:9, 30:45], sharex=ax2, sharey=ax0_hour)
ax_Vehmasmaki = [ax2, ax2_range, ax2_hour]

ax3 = fig.add_subplot(gs[:3, 45:60], sharey=ax0)
ax3_range = fig.add_subplot(gs[3:6, 45:60], sharex=ax3, sharey=ax0_range)
ax3_hour = fig.add_subplot(gs[6:9, 45:60], sharex=ax3, sharey=ax0_hour)
ax_Sodankyla = [ax3, ax3_range, ax3_hour]

ax_cbar = fig.add_subplot(gs_bar[:3])
ax_range_cbar = fig.add_subplot(gs_bar[3:6])
ax_hour_cbar = fig.add_subplot(gs_bar[6:9])
label_holder = 0
for k, ax in zip(location_site, [ax_Uto, ax_Hyytiala, ax_Vehmasmaki, ax_Sodankyla]):
    label_holder += 1
    g = group.get_group(k)
    print(k)

    # Month range median
    dep_mean, month_edge, range_edge, _ = binned_statistic_2d(
        g.datetime.dt.month,
        g['range'],
        g['depo_corrected'],
        bins=[bin_month, bin_range],
        statistic=np.nanmedian)
    # Month range count
    dep_count, month_edge, range_edge = np.histogram2d(
        g.datetime.dt.month,
        g['range'],
        bins=[bin_month, bin_range])

    # mask = (dep_count.T < np.nanpercentile(dep_count, q=10, axis=1).T).T
    # dep_count[mask] = np.nan
    # dep_count[dep_count == 0] = np.nan
    # dep_mean[dep_count < 200] = np.nan
    # dep_mean[np.isnan(dep_count)] = np.nan

    dep_percent = dep_count/np.nansum(dep_count)
    dep_mean[dep_percent < 0.0005] = np.nan

    p = ax[0].pcolormesh(month_edge, range_edge, dep_mean.T, cmap='jet',
                         vmin=0, vmax=0.3)
    cbar = fig.colorbar(p, ax=ax[0], cax=ax_cbar)
    cbar.ax.set_ylabel('$\\delta$')
    ax[0].set_xticklabels([])
    ax[0].yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x/1000:.0f}'))

    p = ax[1].pcolormesh(month_edge, range_edge, dep_percent.T, cmap='viridis',
                         vmin=0, vmax=0.015)
    cbar = fig.colorbar(p, ax=ax[1], cax=ax_range_cbar)
    cbar.ax.set_ylabel('Relative frequency')
    ax[1].set_xticklabels([])
    ax[1].yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x/1000:.0f}'))

    # Month depo
    # H, month_edge, depo_edge = np.histogram2d(
    #     g['month'], g['depo_corrected'],
    #     bins=(bin_month, bin_depo))
    # H_sum = np.sum(H, axis=1)
    # H_plot = H.T/H_sum.T
    # H_plot[H_plot == 0] = np.nan
    # p = ax[2].pcolormesh(X, Y, H_plot*100, cmap='viridis',
    #                   vmin=0, vmax=10)
    # cbar = fig.colorbar(p, ax=ax[2])
    # cbar.ax.set_ylabel('%N')
    # ax[2].set_xticklabels([])

    # Month hour median
    dep_mean, month_edge, time_edge, _ = binned_statistic_2d(
        g.datetime.dt.month,
        g.datetime.dt.hour,
        g['depo_corrected'],
        bins=[bin_month, bin_time],
        statistic=np.nanmedian)

    p = ax[2].pcolormesh(month_edge, time_edge, dep_mean.T,
                         cmap='jet',
                         vmin=1e-5, vmax=0.3)
    cbar = fig.colorbar(p, ax=ax[2], cax=ax_hour_cbar)
    cbar.ax.set_ylabel('$\delta$')

    ax[2].set_xticks(np.arange(1, 13, 3))

    if label_holder > 1:
        for ax_ in ax:
            plt.setp(ax_.get_yticklabels(), visible=False)

for ax_ in [ax0_hour, ax1_hour, ax2_hour, ax3_hour]:
    ax_.set_xticklabels(month_ticklabels)

ax0.set_ylabel('Height a.g.l [km]')
ax0_range.set_ylabel('Height a.g.l [km]')
ax0.set_yticks([0, 1000, 2000])
ax0_range.set_yticks([0, 1000, 2000])
ax0_hour.set_yticks(np.arange(0, 24, 6))
ax0_hour.set_ylabel('Hour')

for ax_ in np.array([ax_Uto, ax_Hyytiala, ax_Vehmasmaki, ax_Sodankyla]).T[:-1].flatten():
    plt.setp(ax_.get_xticklabels(), visible=False)
    # if label_holder > 1:
    #     plt.setp(ax.get_yticklabels(), visible=False)
    #     plt.setp(ax_range.get_yticklabels(), visible=False)
    #     plt.setp(ax_range_count.get_yticklabels(), visible=False)
    #     plt.setp(ax_hour.get_yticklabels(), visible=False)
    #
    # ax_hour.text(-0.0, -0.4, int(key), weight='bold',
    #              transform=ax_hour.transAxes, size=12)

for n, ax in enumerate(np.array([ax_Uto, ax_Hyytiala, ax_Vehmasmaki, ax_Sodankyla]).T.flatten()):
    ax.text(0, 1.05, '(' + string.ascii_lowercase[n] + ')',
            transform=ax.transAxes, size=12)
#

fig.subplots_adjust(hspace=1, wspace=2)
#     for ax in [ax0, ax1, ax2, ax3, ax0_range, ax1_range, ax2_range, ax3_range,
#                ax0_range_count, ax1_range_count, ax2_range_count, ax3_range_count]:
#         plt.setp(ax.get_xticklabels(), visible=False)
#         # ax.tick_params(axis='x',         # changes apply to the x-axis
#         #                which='both',      # both major and minor ticks are affected
#         #                bottom=False,      # ticks along the bottom edge are off
#         #                top=False,         # ticks along the top edge are off
#         #                labelbottom=False)
#
#     for ax in [ax1, ax2, ax3, ax1_range, ax2_range, ax3_range,
#                ax1_range_count, ax2_range_count, ax3_range_count,
#                ax1_hour, ax2_hour, ax3_hour]:
#         ax.tick_params(axis='y',         # changes apply to the x-axis
#                        which='both',      # both major and minor ticks are affected
#                        left=False)
#
#     for ax in [ax1, ax2, ax1_range, ax2_range,
#                ax1_range_count, ax2_range_count,
#                ax1_hour, ax2_hour]:
#         ax.spines['right'].set_color('none')
#         ax.spines['left'].set_color('none')
#
#     for ax in [ax0, ax0_range, ax0_range_count, ax0_hour]:
#         ax.spines['right'].set_color('none')
#
#     for ax in [ax3, ax3_range, ax3_range_count, ax3_hour]:
#         ax.spines['left'].set_color('none')
#
#     if k == 'Sodankyla':
#         for ax in [ax1, ax1_range, ax1_range_count, ax1_hour]:
#             ax.spines['left'].set_color('black')
fig.savefig(path + '/sites/all_sites_summary.png', bbox_inches='tight')
