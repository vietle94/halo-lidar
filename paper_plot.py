import matplotlib.cm as cm
import calendar
from scipy.stats import binned_statistic_2d
import os
from scipy.ndimage import uniform_filter
from sklearn.cluster import DBSCAN
import matplotlib.colors as colors
from scipy.ndimage import maximum_filter
from scipy.ndimage import median_filter
import matplotlib
import copy
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import datetime
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from sklearn.mixture import GaussianMixture
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import glob
import seaborn as sns
import numpy as np
from pathlib import Path
import matplotlib.dates as dates
import halo_data as hd
from matplotlib.colors import LogNorm
import xarray as xr
import string
import scipy.stats as stats
%matplotlib qt

# %%
####################################
# Background SNR
####################################
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
        sd.to_csv('F:/halo/paper/snr_background/' + name + '.csv')


# %%
site_names = ['Uto-32', 'Uto-32XR', 'Hyytiala-33',
              'Hyytiala-46', 'Vehmasmaki-53', 'Sodankyla-54']
fig, axes = plt.subplots(3, 2, figsize=(9, 6), sharex=True, sharey=True)
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

# %%
fig, ax = plt.subplots(figsize=(9, 6), sharex=True, sharey=True)
for file, name in zip(data_list, site_names):
    sd = pd.read_csv(file)
    sd['time'] = pd.to_datetime(sd['time'])
    ax.plot(sd['time'], sd['noise'], '.', label=name,
            markeredgewidth=0.0)
    # ax.set_title(name, weight='bold')
    ax.set_ylim([0, 0.0025])

ax.xaxis.set_major_locator(dates.MonthLocator(6))
ax.legend()
ax.grid()

# %%
save_link = 'F:/halo/paper/snr_background'
site_names = ['Uto-32', 'Uto-32XR', 'Hyytiala-33',
              'Hyytiala-46', 'Vehmasmaki-53', 'Sodankyla-54']
data_list = ['F:/halo/32/depolarization/normal',
             'F:/halo/32/depolarization/xr',
             'F:/halo/33/depolarization',
             'F:/halo/46/depolarization',
             'F:/halo/53/depolarization',
             'F:/halo/54/depolarization']
for site, site_name in zip(data_list, site_names):
    data = hd.getdata(site)
    save_integration_time = {}
    for file in data:
        df = hd.halo_data(file)
        save_integration_time[df.date] = df.integration_time
    s = pd.Series(save_integration_time, name='integration_time')
    s.index.name = 'time'
    s.to_csv('F:/halo/paper/integration_time/' + site_name + '.csv')

# %%
save_link = 'F:/halo/paper/snr_background'
site_names = ['Uto-32', 'Uto-32XR', 'Hyytiala-33',
              'Hyytiala-46', 'Vehmasmaki-53', 'Sodankyla-54']
data_list = ['F:/halo/32/depolarization/normal',
             'F:/halo/32/depolarization/xr',
             'F:/halo/33/depolarization',
             'F:/halo/46/depolarization',
             'F:/halo/53/depolarization',
             'F:/halo/54/depolarization']
for site, site_name in zip(data_list, site_names):
    data = hd.getdata(site)
    save_num_pulses = {}
    for file in data:
        df = hd.halo_data(file)
        save_num_pulses[df.date] = df.info['num_pulses_m1']
    s = pd.Series(save_num_pulses, name='num_pulses')
    s.index.name = 'time'
    s.to_csv('F:/halo/paper/num_pulses/' + site_name + '.csv')

# %%
# Get a list of all files in data folder
snr = 'F:/halo/paper/snr_background/'
integration = 'F:/halo/paper/integration_time/'
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for site in site_names:
    df1 = pd.read_csv(snr + site + '.csv')
    df2 = pd.read_csv(integration + site + '.csv')
    df = df1.merge(df2, how='left')
    df['time'] = pd.to_datetime(df['time'])
    axes[0].plot(df['time'], df['noise'], '.', label=site,
                 markeredgewidth=0.0)
    axes[0].set_ylabel('$\sigma$')
    if site == 'Uto-32XR':
        axes[1].plot(df['time'], df['noise'] * np.sqrt(df['integration_time']*10000),
                     '.', label=site,
                     markeredgewidth=0.0)
    else:
        axes[1].plot(df['time'], df['noise'] * np.sqrt(df['integration_time']*15000),
                     '.', label=site,
                     markeredgewidth=0.0)
    axes[1].set_ylabel('$\sigma$ x $\sqrt{integration\_time * 10000\quador\quad15000}$')
    axes[0].set_ylim([0, 0.0025])
for ax in axes.flatten():
    ax.xaxis.set_major_locator(dates.MonthLocator(6))
    ax.set_xlabel('Time')
    ax.grid()
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=6)
for n, ax in enumerate(axes.flatten()):
    ax.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
            transform=ax.transAxes, size=12)
fig.subplots_adjust(wspace=0.4, bottom=0.2)
fig.savefig('F:/halo/paper/figures/background_snr.png', dpi=150,
            bbox_inches='tight')

# %%
# Get a list of all files in data folder
snr = 'F:/halo/paper/snr_background/'
integration = 'F:/halo/paper/num_pulses/'
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for site in site_names:
    df1 = pd.read_csv(snr + site + '.csv')
    df2 = pd.read_csv(integration + site + '.csv')
    df = df1.merge(df2, how='left')
    df['time'] = pd.to_datetime(df['time'])
    axes[0].plot(df['time'], df['noise'], '.', label=site,
                 markeredgewidth=0.0)
    axes[0].set_ylabel('$\sigma$')
    axes[1].plot(df['time'], df['noise'] * np.sqrt(df['num_pulses']),
                 '.', label=site,
                 markeredgewidth=0.0)
    axes[1].set_ylabel('$\sigma$ x $\sqrt{number\quad of\quadpulses}$')
    axes[0].set_ylim([0, 0.0025])
for ax in axes.flatten():
    ax.xaxis.set_major_locator(dates.MonthLocator(6))
    ax.set_xlabel('Time')
    ax.grid()
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=6)
for n, ax in enumerate(axes.flatten()):
    ax.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
            transform=ax.transAxes, size=12)
fig.subplots_adjust(wspace=0.4, bottom=0.2)
fig.savefig('F:/halo/paper/figures/background_snr_num.png', dpi=150,
            bbox_inches='tight')

# %%
####################################
# Depo cloudbase
####################################
site_names = ['Uto-32', 'Uto-32XR', 'Hyytiala-33',
              'Hyytiala-46', 'Vehmasmaki-53', 'Sodankyla-54']
fig, axes = plt.subplots(3, 2, figsize=(9, 6), sharex=True)
table = pd.read_csv('F:/halo/paper/depo_cloudbase/result.csv')
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
mean = []
median = []
std = []
id = []
x_axis = np.linspace(-0.01, 0.1, 100)
for (systemID, group), ax, name in zip(table.groupby('systemID'),
                                       axes.flatten()[1:],
                                       site_names[1:]):
    if systemID == 32:
        x = group['depo'][(group['depo'] > -0.01) & (group['depo'] < 0.1) & (pd.to_datetime(
            group[['year', 'month', 'day']]) < '2017-11-22')]
        y_, x_, _ = axes[0, 0].hist(x, bins=80)
        x_ = (x_[1:]+x_[:-1])/2
        textstr = '\n'.join((
            r'$\mu=%.3f$' % (np.mean(x), ),
            r'$\mathrm{median}=%.3f$' % (np.median(x), ),
            r'$\sigma=%.3f$' % (np.std(x), )))
        mean.append(np.mean(x))
        std.append(np.std(x))
        id.append(systemID)
        median.append(np.median(x))
        axes[0, 0].text(0.6, 0.95, textstr, transform=axes[0, 0].transAxes,
                        verticalalignment='top', bbox=props)

        # gmm = GaussianMixture(n_components=2, max_iter=1000)
        # gmm.fit(x.values.reshape(-1, 1))
        # smean = gmm.means_.ravel()
        # sstd = np.sqrt(gmm.covariances_).ravel()
        # sweight = np.sqrt(gmm.weights_).ravel()
        # sort_idx = np.argsort(smean)
        # smean = smean[sort_idx]
        # sstd = sstd[sort_idx]
        # sweight = sweight[sort_idx]
        # axes[0, 0].set_title('Distribution of depo at cloud base, ' + f'\n\
        # left peak is {smean[0]:.4f} $\pm$ {sstd[0]:.4f}', weight='bold')
        # y = stats.norm.pdf(x_axis, smean[0], sstd[0])
        # axes[0, 0].plot(x_axis, y)
        #
        # y1 = stats.norm.pdf(x_axis, smean[1], sstd[1])
        # axes[0, 0].plot(x_axis, y1)
        expected = (0.01055434, 0.00694231, 250, 0.04292838, 0.01806932, 125)
        params, cov = curve_fit(bimodal, x_, y_, expected)
        sigma = np.sqrt(np.diag(cov))
        axes[0, 0].plot(x_, bimodal(x_, *params))
        axes[0, 0].set_title(f'Left peak is {params[0]:.3f} $\pm$ {params[1]:.3f}')

        x = group['depo'][(group['depo'] > -0.01) & (group['depo'] < 0.1) & (pd.to_datetime(
            group[['year', 'month', 'day']]) >= '2017-11-22')]
        y_, x_, _ = axes[0, 1].hist(x, bins=100)
        x_ = (x_[1:]+x_[:-1])/2
        textstr = '\n'.join((
            r'$\mu=%.3f$' % (np.mean(x), ),
            r'$\mathrm{median}=%.3f$' % (np.median(x), ),
            r'$\sigma=%.3f$' % (np.std(x), )))

        axes[0, 1].text(0.6, 0.95, textstr, transform=axes[0, 1].transAxes,
                        verticalalignment='top', bbox=props)
        expected = (0.01055434, 0.00694231, 250)
        params, cov = curve_fit(gauss, x_, y_, expected)
        sigma = np.sqrt(np.diag(cov))
        axes[0, 1].plot(x_, gauss(x_, *params))
        axes[0, 1].set_title(f'Peak is {params[0]:.3f} $\pm$ {params[1]:.3f}')

        mean.append(np.mean(x))
        std.append(np.std(x))
        id.append(systemID)
        median.append(np.median(x))
        # axes[0, 1].set_title('Uto-32XR', weight='bold')
    elif systemID == 46:
        x = group['depo'][(group['depo'] > -0.01) & (group['depo'] < 0.1)]
        y_, x_, _ = ax.hist(x, bins=80)
        x_ = (x_[1:]+x_[:-1])/2
        textstr = '\n'.join((
            r'$\mu=%.3f$' % (np.mean(x), ),
            r'$\mathrm{median}=%.3f$' % (np.median(x), ),
            r'$\sigma=%.3f$' % (np.std(x), )))
        mean.append(np.mean(x))
        std.append(np.std(x))
        id.append(systemID)
        median.append(np.median(x))
        ax.text(0.6, 0.95, textstr, transform=ax.transAxes,
                verticalalignment='top', bbox=props)

        # gmm = GaussianMixture(n_components=2, max_iter=1000)
        # gmm.fit(x.values.reshape(-1, 1))
        # smean = gmm.means_.ravel()
        # sstd = np.sqrt(gmm.covariances_).ravel()
        # sweight = np.sqrt(gmm.weights_).ravel()
        # sort_idx = np.argsort(smean)
        # smean = smean[sort_idx]
        # sstd = sstd[sort_idx]
        # sweight = sweight[sort_idx]
        # ax.set_title('Distribution of depo at cloud base, ' + f'\n\
        # left peak is {smean[0]:.4f} $\pm$ {sstd[0]:.4f}', weight='bold')
        # y = stats.norm.pdf(x_axis, smean[0], sstd[0])
        # ax.plot(x_axis, y)
        #
        # y1 = stats.norm.pdf(x_axis, smean[1], sstd[1])
        # ax.plot(x_axis, y1)
        expected = (0.01055434, 0.00694231, 250, 0.04292838, 0.01806932, 125)
        params, cov = curve_fit(bimodal, x_, y_, expected)
        sigma = np.sqrt(np.diag(cov))
        ax.plot(x_, bimodal(x_, *params))
        ax.set_title(f'Left peak is {params[0]:.3f} $\pm$ {params[1]:.3f}')
    else:
        x = group['depo'][(group['depo'] > -0.01) & (group['depo'] < 0.1)]
        y_, x_, _ = ax.hist(x, bins=80)
        x_ = (x_[1:]+x_[:-1])/2

        textstr = '\n'.join((
            r'$\mu=%.3f$' % (np.mean(x), ),
            r'$\mathrm{median}=%.3f$' % (np.median(x), ),
            r'$\sigma=%.3f$' % (np.std(x), )))

        ax.text(0.6, 0.95, textstr, transform=ax.transAxes,
                verticalalignment='top', bbox=props)
        y = stats.norm.pdf(x_axis, np.median(x[x < 0.02]), np.std(x[x < 0.02]))
        # ax.plot(x_axis, y)
        # ax.set_title(name, weight='bold')
        expected = (0.01055434, 0.00694231, 250)
        params, cov = curve_fit(gauss, x_, y_, expected)
        sigma = np.sqrt(np.diag(cov))
        ax.plot(x_, gauss(x_, *params))
        ax.set_title(f'Peak is {params[0]:.3f} $\pm$ {params[1]:.3f}')
        mean.append(np.mean(x))
        std.append(np.std(x))
        id.append(systemID)
        median.append(np.median(x))
for n, ax in enumerate(axes.flatten()):
    ax.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
            transform=ax.transAxes, size=12)
    ax.set_ylabel('N')
axes.flatten()[-2].set_xlabel('$\delta$')
axes.flatten()[-1].set_xlabel('$\delta$')
fig.subplots_adjust(hspace=0.5, wspace=0.3)
fig.savefig('F:/halo/paper/figures/depo_cloudbase2.png', dpi=150,
            bbox_inches='tight')

# %%
site_names = ['Uto-32', 'Uto-32XR', 'Hyytiala-33',
              'Hyytiala-46', 'Vehmasmaki-53', 'Sodankyla-54']
fig, axes = plt.subplots(3, 2, figsize=(9, 6), sharex=True)
table = pd.read_csv('F:/halo/paper/depo_cloudbase/result.csv')
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
mean = []
median = []
std = []
id = []
x_axis = np.linspace(-0.01, 0.1, 100)
for (systemID, group), ax, name in zip(table.groupby('systemID'),
                                       axes.flatten()[1:],
                                       site_names[1:]):
    if systemID == 32:
        x = group[(pd.to_datetime(
            group[['year', 'month', 'day']]) < '2017-11-22')]
        x = (x['cross_signal'] - 1)/(x['co_signal'] - 1)
        y_, x_, _ = axes[0, 0].hist(x, bins=80)
        x_ = (x_[1:]+x_[:-1])/2
        textstr = '\n'.join((
            r'$\mu=%.3f$' % (np.mean(x), ),
            r'$\mathrm{median}=%.3f$' % (np.median(x), ),
            r'$\sigma=%.3f$' % (np.std(x), )))
        mean.append(np.mean(x))
        std.append(np.std(x))
        id.append(systemID)
        median.append(np.median(x))
        axes[0, 0].text(0.6, 0.95, textstr, transform=axes[0, 0].transAxes,
                        verticalalignment='top', bbox=props)

        # gmm = GaussianMixture(n_components=2, max_iter=1000)
        # gmm.fit(x.values.reshape(-1, 1))
        # smean = gmm.means_.ravel()
        # sstd = np.sqrt(gmm.covariances_).ravel()
        # sweight = np.sqrt(gmm.weights_).ravel()
        # sort_idx = np.argsort(smean)
        # smean = smean[sort_idx]
        # sstd = sstd[sort_idx]
        # sweight = sweight[sort_idx]
        # axes[0, 0].set_title('Distribution of depo at cloud base, ' + f'\n\
        # left peak is {smean[0]:.4f} $\pm$ {sstd[0]:.4f}', weight='bold')
        # y = stats.norm.pdf(x_axis, smean[0], sstd[0])
        # axes[0, 0].plot(x_axis, y)
        #
        # y1 = stats.norm.pdf(x_axis, smean[1], sstd[1])
        # axes[0, 0].plot(x_axis, y1)
        expected = (0.01055434, 0.00694231, 250, 0.04292838, 0.01806932, 125)
        params, cov = curve_fit(bimodal, x_, y_, expected)
        sigma = np.sqrt(np.diag(cov))
        axes[0, 0].plot(x_, bimodal(x_, *params))
        axes[0, 0].set_title(f'Left peak is {params[0]:.3f} $\pm$ {params[1]:.3f}')

        x = group[(pd.to_datetime(
            group[['year', 'month', 'day']]) >= '2017-11-22')]
        x = (x['cross_signal'] - 1)/(x['co_signal'] - 1)

        y_, x_, _ = axes[0, 1].hist(x, bins=100)
        x_ = (x_[1:]+x_[:-1])/2
        textstr = '\n'.join((
            r'$\mu=%.3f$' % (np.mean(x), ),
            r'$\mathrm{median}=%.3f$' % (np.median(x), ),
            r'$\sigma=%.3f$' % (np.std(x), )))

        axes[0, 1].text(0.6, 0.95, textstr, transform=axes[0, 1].transAxes,
                        verticalalignment='top', bbox=props)
        expected = (0.01055434, 0.00694231, 250)
        params, cov = curve_fit(gauss, x_, y_, expected)
        sigma = np.sqrt(np.diag(cov))
        axes[0, 1].plot(x_, gauss(x_, *params))
        axes[0, 1].set_title(f'Peak is {params[0]:.3f} $\pm$ {params[1]:.3f}')

        mean.append(np.mean(x))
        std.append(np.std(x))
        id.append(systemID)
        median.append(np.median(x))
        # axes[0, 1].set_title('Uto-32XR', weight='bold')
    elif systemID == 46:
        x = group
        x = (x['cross_signal'] - 1)/(x['co_signal'] - 1)

        y_, x_, _ = ax.hist(x, bins=80)
        x_ = (x_[1:]+x_[:-1])/2
        textstr = '\n'.join((
            r'$\mu=%.3f$' % (np.mean(x), ),
            r'$\mathrm{median}=%.3f$' % (np.median(x), ),
            r'$\sigma=%.3f$' % (np.std(x), )))
        mean.append(np.mean(x))
        std.append(np.std(x))
        id.append(systemID)
        median.append(np.median(x))
        ax.text(0.6, 0.95, textstr, transform=ax.transAxes,
                verticalalignment='top', bbox=props)

        # gmm = GaussianMixture(n_components=2, max_iter=1000)
        # gmm.fit(x.values.reshape(-1, 1))
        # smean = gmm.means_.ravel()
        # sstd = np.sqrt(gmm.covariances_).ravel()
        # sweight = np.sqrt(gmm.weights_).ravel()
        # sort_idx = np.argsort(smean)
        # smean = smean[sort_idx]
        # sstd = sstd[sort_idx]
        # sweight = sweight[sort_idx]
        # ax.set_title('Distribution of depo at cloud base, ' + f'\n\
        # left peak is {smean[0]:.4f} $\pm$ {sstd[0]:.4f}', weight='bold')
        # y = stats.norm.pdf(x_axis, smean[0], sstd[0])
        # ax.plot(x_axis, y)
        #
        # y1 = stats.norm.pdf(x_axis, smean[1], sstd[1])
        # ax.plot(x_axis, y1)
        expected = (0.01055434, 0.00694231, 250, 0.04292838, 0.01806932, 125)
        params, cov = curve_fit(bimodal, x_, y_, expected)
        sigma = np.sqrt(np.diag(cov))
        ax.plot(x_, bimodal(x_, *params))
        ax.set_title(f'Left peak is {params[0]:.3f} $\pm$ {params[1]:.3f}')
    else:
        x = group
        x = (x['cross_signal'] - 1)/(x['co_signal'] - 1)

        y_, x_, _ = ax.hist(x, bins=80)
        x_ = (x_[1:]+x_[:-1])/2

        textstr = '\n'.join((
            r'$\mu=%.3f$' % (np.mean(x), ),
            r'$\mathrm{median}=%.3f$' % (np.median(x), ),
            r'$\sigma=%.3f$' % (np.std(x), )))

        ax.text(0.6, 0.95, textstr, transform=ax.transAxes,
                verticalalignment='top', bbox=props)
        y = stats.norm.pdf(x_axis, np.median(x[x < 0.02]), np.std(x[x < 0.02]))
        # ax.plot(x_axis, y)
        # ax.set_title(name, weight='bold')
        expected = (0.01055434, 0.00694231, 250)
        params, cov = curve_fit(gauss, x_, y_, expected)
        sigma = np.sqrt(np.diag(cov))
        ax.plot(x_, gauss(x_, *params))
        ax.set_title(f'Peak is {params[0]:.3f} $\pm$ {params[1]:.3f}')
        mean.append(np.mean(x))
        std.append(np.std(x))
        id.append(systemID)
        median.append(np.median(x))
for n, ax in enumerate(axes.flatten()):
    ax.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
            transform=ax.transAxes, size=12)
    ax.set_ylabel('N')
axes.flatten()[-2].set_xlabel('$\delta$')
axes.flatten()[-1].set_xlabel('$\delta$')
fig.subplots_adjust(hspace=0.5, wspace=0.3)
# fig.savefig('F:/halo/paper/figures/depo_cloudbase2.png', dpi=150,
#             bbox_inches='tight')


# %%


def gauss(x, mu, sigma, A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)


def bimodal(x, mu1, sigma1, A1, mu2, sigma2, A2):
    return gauss(x, mu1, sigma1, A1)+gauss(x, mu2, sigma2, A2)


# %%
# i_plot = 119216
i_plot = 235144
site = str(int(table.iloc[i_plot]['systemID']))
if site == '32':
    data = hd.getdata('F:/halo/' + site + '/depolarization/normal') + \
        hd.getdata('F:/halo/' + site + '/depolarization/xr')
else:
    data = hd.getdata('F:/halo/' + site + '/depolarization/')
date = ''.join(list(map(lambda x: str(int(table.iloc[i_plot][x])).zfill(2),
                        ['year', 'month', 'day'])))
file = [file for file in data if date in file][0]

df = hd.halo_data(file)
df.filter_height()
df.unmask999()
df.filter(variables=['beta_raw', 'depo_raw'],
          ref='co_signal',
          threshold=1 + 3 * df.snr_sd)

# mask_range_plot = (df.data['range'] <= table.iloc[i_plot]['range'] + 100) & (
#     df.data['range'] >= table.iloc[i_plot]['range'] - 100)
cloud_base_height = table.iloc[i_plot]['range']
mask_range_plot = (df.data['range'] <= cloud_base_height + 300)
mask_time_plot = df.data['time'] == table.iloc[i_plot]['time']
depo_profile_plot = df.data['depo_raw'][mask_time_plot,
                                        mask_range_plot]
co_signal_profile_plot = df.data['co_signal'][mask_time_plot,
                                              mask_range_plot]
beta_profile_plot = df.data['beta_raw'][mask_time_plot,
                                        mask_range_plot]

fig, axes = plt.subplots(1, 3, figsize=(9, 4), sharey=True)

for ax, var, lab in zip(axes.flatten(), ['depo_raw', 'co_signal', 'beta_raw'],
                        ['$\delta$', '$SNR_{co}$', r'$\beta\quad[Mm^{-1}]$']):
    for h, leg in zip([df.data['range'] <= cloud_base_height,
                       df.data['range'] == cloud_base_height,
                       (cloud_base_height < df.data['range']) &
                       (df.data['range'] <= cloud_base_height + 300)],
                      ['Aerosol', 'Cloud base', 'In-cloud']):
        ax.plot(df.data[var][mask_time_plot, h],
                df.data['range'][h], '.', label=leg)
    ax.grid()
    # ax.axhline(y=cloud_base_height, linestyle='--', color='grey')
    ax.set_xlabel(lab)
axes[0].set_xlim([-0.05, 0.1])
axes[0].set_ylabel('Height [m]')
ax.set_xscale('log')
for n, ax in enumerate(axes.flatten()):
    ax.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
            transform=ax.transAxes, size=12)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3)
fig.subplots_adjust(bottom=0.2)
fig.savefig('F:/halo/paper/figures/' + df.filename + '_depo_profile.png')

# %%
table32 = table[(table['systemID'] == 32) & (pd.to_datetime(
    table[['year', 'month', 'day']]) >= '2017-11-22')]

co_cross_data = table32[['co_signal', 'cross_signal']].dropna()
H, co_edges, cross_edges = np.histogram2d(
    co_cross_data['co_signal'] - 1,
    co_cross_data['cross_signal'] - 1,
    bins=500)
X, Y = np.meshgrid(co_edges, cross_edges)
fig, ax = plt.subplots(figsize=(6, 3))
p = ax.pcolormesh(X, Y, H.T, norm=LogNorm())
ax.set_xlabel('$SNR_{co}$')
ax.set_ylabel('$SNR_{cross}$')
colorbar = fig.colorbar(p, ax=ax)
colorbar.ax.set_ylabel('N')
ax.plot(co_cross_data['co_signal'] - 1,
        (co_cross_data['co_signal'] - 1) * 0.004,
        label=r'$SNR_{cross} = 0.004SNR_{co}$',
        linewidth=0.5)
ax.legend(loc='upper left')
fig.savefig('F:/halo/paper/figures/co_cross_saturation.png', dpi=150,
            bbox_inches='tight')

# %%


def cantrell_fit(x, y, wx, wy, m_ini, r, tol):
    n = len(x)
    m1 = m_ini
    a = np.sqrt(wx * wy)
    m2 = m1+1
    print(np.abs(m1-m2))
    print(tol)
    print(np.abs(m1-m2) > tol)
    while (np.abs(m1-m2) > tol):
        m1 = m2
        W = (wx * wy)/(wx + (m1**2*wy - 2*m1*r*a))

        x_bar = (W*x)/np.sum(W)
        y_bar = (W*y)/np.sum(W)
        U = x - x_bar
        V = y - y_bar
        beta = W * (U/wy + (m1*V)/wx - (m1*U + V)*(r/a))
        m2 = ((W*beta)*V)/((W*beta)*U)

    m = m2
    b = y_bar - m*x_bar
    S = W*(y - (m*x + b))**2
    gof = S/(n-2)
    beta_bar = (W*beta)/np.sum(W)
    s2_m = 1/(W*((beta - beta_bar)**2))
    s2_b = 1/(np.sum(W)) + (x_bar + beta_bar)**2 * s2_m
    s_b = np.sqrt(s2_b)*np.sqrt(gof)
    s_m = np.sqrt(s2_m)*np.sqrt(gof)

    return m, b, s_m, s_b, gof


# %%
xedges = np.linspace(0, 8, 500)
yedges = np.linspace(0, 0.2, 500)
fig, axes = plt.subplots(3, 2, figsize=(16, 9), sharex=True, sharey=True)
for (systemID, group), ax, name in zip(table.groupby('systemID'),
                                       axes.flatten()[1:],
                                       site_names[1:]):
    if systemID == 32:
        x = group[(pd.to_datetime(
            group[['year', 'month', 'day']]) < '2017-11-22')]
        co_cross_data = x[['co_signal', 'cross_signal']].dropna()

        m, b, s_m, s_b, gof = cantrell_fit(
            co_cross_data['co_signal']-1,
            co_cross_data['cross_signal']-1,
            np.repeat(1/np.var(co_cross_data['co_signal']-1),
                      len(co_cross_data['co_signal'])),
            np.repeat(1/np.var(co_cross_data['cross_signal']-1),
                      len(co_cross_data['cross_signal'])),
            0.03, 0, 0.001)
        reg = LinearRegression().fit((co_cross_data['co_signal']-1).values.reshape(-1, 1),
                                     (co_cross_data['cross_signal']-1).values.reshape(-1, 1))

        H, co_edges, cross_edges = np.histogram2d(
            co_cross_data['co_signal'] - 1,
            co_cross_data['cross_signal'] - 1,
            bins=500)
        X, Y = np.meshgrid(co_edges, cross_edges)
        p = axes[0, 0].pcolormesh(X, Y, H.T, norm=LogNorm())
        axes[0, 0].set_xlabel('$SNR_{co}$')
        axes[0, 0].set_ylabel('$SNR_{cross}$')
        colorbar = fig.colorbar(p, ax=axes[0, 0])
        colorbar.ax.set_ylabel('N')
        axes[0, 0].set_title('Uto-32' + f' with gof = {gof:.3f}')
        axes[0, 0].plot(co_cross_data['co_signal'] - 1,
                        (co_cross_data['co_signal'] - 1) * m + b,
                        label=fr'$cross\_SNR = {m:.3f} * co\_SNR {b:.3f}$',
                        linewidth=0.5)
        axes[0, 0].plot(co_cross_data['co_signal'] - 1,
                        (co_cross_data['co_signal'] - 1) *
                        reg.coef_.flatten()[0] + reg.intercept_.flatten()[0],
                        label=fr'$cross\_SNR = {reg.coef_.flatten()[0]:.3f} * co\_SNR$ {reg.intercept_.flatten()[0]:.3f}',
                        linewidth=0.5)
        axes[0, 0].legend(loc='upper left')
        print(np.min((co_cross_data['cross_signal']-1)))

        x = group[(pd.to_datetime(
            group[['year', 'month', 'day']]) >= '2017-11-22')]
        co_cross_data = x[['co_signal', 'cross_signal']].dropna()
        m, b, s_m, s_b, gof = cantrell_fit(
            co_cross_data['co_signal']-1,
            co_cross_data['cross_signal']-1,
            np.repeat(1/np.var(co_cross_data['co_signal']-1),
                      len(co_cross_data['co_signal'])),
            np.repeat(1/np.var(co_cross_data['cross_signal']-1),
                      len(co_cross_data['cross_signal'])),
            0.03, 0, 0.001)
        reg = LinearRegression().fit((co_cross_data['co_signal']-1).values.reshape(-1, 1),
                                     (co_cross_data['cross_signal']-1).values.reshape(-1, 1))
        H, co_edges, cross_edges = np.histogram2d(
            co_cross_data['co_signal'] - 1,
            co_cross_data['cross_signal'] - 1,
            bins=500)
        X, Y = np.meshgrid(co_edges, cross_edges)
        p = axes[0, 1].pcolormesh(X, Y, H.T, norm=LogNorm())
        axes[0, 1].set_xlabel('$SNR_{co}$')
        axes[0, 1].set_ylabel('$SNR_{cross}$')
        axes[0, 1].set_title(name + f' with gof = {gof:.3f}')
        colorbar = fig.colorbar(p, ax=axes[0, 1])
        colorbar.ax.set_ylabel('N')
        axes[0, 1].plot(co_cross_data['co_signal'] - 1,
                        (co_cross_data['co_signal'] - 1) * m + b,
                        label=fr'$cross\_SNR = {m:.3f} * co\_SNR {b:.3f}$',
                        linewidth=0.5)
        axes[0, 1].plot(co_cross_data['co_signal'] - 1,
                        (co_cross_data['co_signal'] - 1) *
                        reg.coef_.flatten()[0] + reg.intercept_.flatten()[0],
                        label=fr'$cross\_SNR = {reg.coef_.flatten()[0]:.3f} * co\_SNR$ {reg.intercept_.flatten()[0]:.3f}',
                        linewidth=0.5)
        axes[0, 1].legend(loc='upper left')
        print(np.min((co_cross_data['cross_signal']-1)))

    else:
        co_cross_data = group[['co_signal', 'cross_signal']].dropna()
        m, b, s_m, s_b, gof = cantrell_fit(
            co_cross_data['co_signal']-1,
            co_cross_data['cross_signal']-1,
            np.repeat(1/np.var(co_cross_data['co_signal']-1),
                      len(co_cross_data['co_signal'])),
            np.repeat(1/np.var(co_cross_data['cross_signal']-1),
                      len(co_cross_data['cross_signal'])),
            0.03, 0, 0.001)
        reg = LinearRegression().fit((co_cross_data['co_signal']-1).values.reshape(-1, 1),
                                     (co_cross_data['cross_signal']-1).values.reshape(-1, 1))
        H, co_edges, cross_edges = np.histogram2d(
            co_cross_data['co_signal'] - 1,
            co_cross_data['cross_signal'] - 1,
            bins=500)
        X, Y = np.meshgrid(co_edges, cross_edges)
        p = ax.pcolormesh(X, Y, H.T, norm=LogNorm())
        ax.set_xlabel('$SNR_{co}$')
        ax.set_ylabel('$SNR_{cross}$')
        ax.set_title(name + f' with gof = {gof:.3f}')
        colorbar = fig.colorbar(p, ax=ax)
        colorbar.ax.set_ylabel('N')
        ax.plot(co_cross_data['co_signal'] - 1,
                (co_cross_data['co_signal'] - 1) * m + b,
                label=fr'$cross\_SNR = {m:.3f} * co\_SNR {b:.3f}$',
                linewidth=0.5)
        ax.plot(co_cross_data['co_signal'] - 1,
                (co_cross_data['co_signal'] - 1) *
                reg.coef_.flatten()[0] + reg.intercept_.flatten()[0],
                label=fr'$cross\_SNR = {reg.coef_.flatten()[0]:.3f} * co\_SNR$ {reg.intercept_.flatten()[0]:.3f}',
                linewidth=0.5)
        ax.legend(loc='upper left')
        print(np.min((co_cross_data['cross_signal']-1)))
for ax in axes.flatten():
    ax.set_xlim([-0.1, 8])
    ax.set_ylim([-0.01, 0.2])
fig.savefig('co_cross.png', bbox_inches='tight')
# %%


def cantrell_fit(x, y, wx, wy, m_ini, r, tol):
    n = len(x)
    m1 = m_ini
    a = np.sqrt(wx * wy)
    m2 = m1+1
    while (np.abs(m1-m2) > tol):
        m1 = m2
        W = (wx * wy)/(wx + (m1**2*wy - 2*m1*r*a))

        x_bar = np.sum(W*x)/np.sum(W)
        y_bar = np.sum(W*y)/np.sum(W)
        U = x - x_bar
        V = y - y_bar
        beta = W * (U/wy + (m1*V)/wx - (m1*U + V)*(r/a))
        m2 = np.sum((W*beta)*V)/np.sum((W*beta)*U)
    m = m2
    b = y_bar - m*x_bar
    # S = np.sum(W*(y - (m*x + b))**2)
    S = np.sum(W*(y - (m*x + b))**2)
    # S_ = np.sum((y - (m*x + b))**2)
    # # print(S_/(n-2))
    gof = S/(n-2)
    beta_bar = np.sum(W*beta)/np.sum(W)
    s2_m = 1/np.sum(W*((beta - beta_bar)**2))
    s2_b = 1/(np.sum(W)) + (x_bar + beta_bar)**2 * s2_m
    s_b = np.sqrt(s2_b)*np.sqrt(gof)
    s_m = np.sqrt(s2_m)*np.sqrt(gof)

    return m, b, s_m, s_b, gof


###########################################
# %%
###########################################
data = hd.getdata('F:/halo/46/depolarization')

# %%
date = '20180812'
file = [file for file in data if date in file][0]
df = hd.halo_data(file)

df.filter_height()
df.unmask999()
df.depo_cross_adj()

# %%
m_ = ((df.data['time'] > 5.70) &
      (df.data['time'] < 5.85)) | ((df.data['time'] > 6.20) &
                                   (df.data['time'] < 6.6))
m_ = m_ | ((df.data['time'] > 17.7) &
           (df.data['time'] < 17.9)) | ((df.data['time'] > 7.02) &
                                        (df.data['time'] < 7.1))

temp_co = df.data['co_signal'].copy()
lol = np.isnan(np.log10(df.data['beta_raw']))
lol[~m_, :] = False
temp_co[lol] = np.nan


# %%
fig, axes = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(6, 5))
p = axes[0].pcolormesh(df.data['time'], df.data['range'],
                       np.log10(df.data['beta_raw']).T, cmap='jet',
                       vmin=-8, vmax=-4)
axes[1].yaxis.set_major_formatter(hd.m_km_ticks())
axes[0].set_xlim([0, 24])
cbar = fig.colorbar(p, ax=axes[0], fraction=0.05)
cbar.ax.set_ylabel(r'$\beta\quad[Mm^{-1}]$', rotation=90)
axes[0].set_ylabel('Height [km]')
p = axes[1].pcolormesh(df.data['time'], df.data['range'],
                       df.data['v_raw'].T, cmap='jet',
                       vmin=-2, vmax=2)
cbar = fig.colorbar(p, ax=axes[1], fraction=0.05)
cbar.ax.set_ylabel('v [' + df.units.get('v_raw', None) + ']', rotation=90)
axes[1].set_ylabel('Height [km]')

p = axes[2].pcolormesh(df.data['time'], df.data['range'],
                       temp_co.T - 1, cmap='jet',
                       vmin=0.995 - 1, vmax=1.005 - 1)
cbar = fig.colorbar(p, ax=axes[2], fraction=0.05)
cbar.ax.set_ylabel('$SNR_{co}$', rotation=90)
axes[2].set_ylabel('Height [km]')
axes[2].set_xlabel('Time [UTC - hour]')
fig.tight_layout()
fig.savefig('F:/halo/paper/figures/pre-processed.png', dpi=150,
            bbox_inches='tight')

# %%
df.filter(variables=['beta_raw'],
          ref='co_signal',
          threshold=1 + df.snr_sd)

# Save before further filtering
beta_save = df.data['beta_raw'].flatten()
depo_save = df.data['depo_adj'].flatten()
co_save = df.data['co_signal'].flatten()
cross_save = df.data['cross_signal'].flatten()  # Already adjusted with bleed

df.data['classifier'] = np.zeros(df.data['beta_raw'].shape, dtype=int)

log_beta = np.log10(df.data['beta_raw'])
# Aerosol
aerosol = log_beta < -5.5
class1 = df.data['classifier'].copy()
class1[aerosol] = 10
# Small size median filter to remove noise
aerosol_smoothed = median_filter(aerosol, size=11)
# Remove thin bridges, better for the clustering
aerosol_smoothed = median_filter(aerosol_smoothed, size=(15, 1))

# %%
df.data['classifier'][aerosol_smoothed] = 10
class2 = df.data['classifier'].copy()
# %%
# cmap = mpl.colors.ListedColormap(
#     ['white', '#2ca02c', 'blue', 'red', 'gray'])
# boundaries = [0, 10, 20, 30, 40, 50]
# norm = mpl.colors.BoundaryNorm(boundaries, cmap.N, clip=True)
# fig, ax = plt.subplots(figsize=(6, 2))
# p = ax.pcolormesh(df.data['time'], df.data['range'],
#                   df.data['classifier'].T,
#                   cmap=cmap, norm=norm)
# ax.yaxis.set_major_formatter(hd.m_km_ticks())
# ax.set_ylabel('Range [km, a.g.l]')
# ax.set_xlabel('Time [UTC - hour]')
# cbar = fig.colorbar(p, ax=ax, ticks=[5, 15, 25, 35, 45])
# cbar.ax.set_yticklabels(['Background', 'Aerosol',
#                          'Precipitation', 'Clouds', 'Undefined'])
# fig.tight_layout()
# fig.savefig(path + '/algorithm_aerosol.png', bbox_inches='tight')

# %%
df.filter(variables=['beta_raw', 'v_raw', 'depo_adj'],
          ref='co_signal',
          threshold=1 + 3 * df.snr_sd)
log_beta = np.log10(df.data['beta_raw'])

range_save = np.tile(df.data['range'],
                     df.data['beta_raw'].shape[0])

time_save = np.repeat(df.data['time'],
                      df.data['beta_raw'].shape[1])
v_save = df.data['v_raw'].flatten()  # put here to avoid noisy values at 1sd snr

# Liquid
liquid = log_beta > -5.5
class3 = df.data['classifier'].copy()
class3[liquid] = 30

# maximum filter to increase the size of liquid region
liquid_max = maximum_filter(liquid, size=5)
# Median filter to remove background noise
liquid_smoothed = median_filter(liquid_max, size=13)

df.data['classifier'][liquid_smoothed] = 30
class4 = df.data['classifier'].copy()

# %%
fig, axes = plt.subplots(2, 2, figsize=(10, 4))
for ax, x in zip(axes.flatten(), [class1, class2, class3, class4]):
    p = ax.pcolormesh(df.data['time'], df.data['range'],
                      x.T,
                      cmap=cmap, norm=norm)
    ax.yaxis.set_major_formatter(hd.m_km_ticks())
    ax.set_ylabel('Height [km]')
    ax.set_xlabel('Time [UTC - hour]')
    cbar = fig.colorbar(p, ax=ax, ticks=[5, 15, 25, 35, 45])
    cbar.ax.set_yticklabels(['Background', 'Aerosol',
                             'Precipitation', 'Clouds', 'Undefined'])
for n, ax in enumerate(axes.flatten()):
    ax.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
            transform=ax.transAxes, size=12)
fig.tight_layout()
fig.savefig('F:/halo/paper/figures/algorithm_cloud_aerosol.png', bbox_inches='tight')

# %%
# updraft - indication of aerosol zone
updraft = df.data['v_raw'] > 1
updraft_smooth = median_filter(updraft, size=3)
updraft_max = maximum_filter(updraft_smooth, size=91)

# Fill the gap in aerosol zone
updraft_median = median_filter(updraft_max, size=31)

# precipitation < -1 (center of precipitation)
precipitation_1 = (log_beta > -7) & (df.data['v_raw'] < -1)
class5 = df.data['classifier'].copy()
class5[precipitation_1] = 20
precipitation_1_median = median_filter(precipitation_1, size=9)

# Only select precipitation outside of aerosol zone
precipitation_1_ne = precipitation_1_median * ~updraft_median
precipitation_1_median_smooth = median_filter(precipitation_1_ne,
                                              size=3)
precipitation = precipitation_1_median_smooth
class6 = df.data['classifier'].copy()
class6[precipitation] = 20

# precipitation < -0.5 (include all precipitation)
precipitation_1_low = (log_beta > -7) & (df.data['v_raw'] < -0.5)

class7 = df.data['classifier'].copy()
class7[precipitation_1_low] = 20
# Avoid ebola infection surrounding updraft
# Useful to contain error during ebola precipitation
updraft_ebola = df.data['v_raw'] > 0.2
updraft_ebola_max = maximum_filter(updraft_ebola, size=3)

# %%
# fig, ax = plt.subplots(2, 1, figsize=(6, 4), sharex=True, sharey=True)
temp = df.data['classifier'].copy()

# Ebola precipitation
for i in range(1500):
    if i == 1:
        temp[precipitation] = 20
        # p = ax[0].pcolormesh(df.data['time'], df.data['range'],
        #                      temp.T, cmap=cmap, norm=norm)
        # ax[0].yaxis.set_major_formatter(hd.m_km_ticks())
        # ax[0].set_ylabel('Range [km, a.g.l]')
        # cbar = fig.colorbar(p, ax=ax[0], ticks=[5, 15, 25, 35, 45])
        # cbar.ax.set_yticklabels(['Background', 'Aerosol',
        #                          'Precipitation', 'Clouds', 'Undefined'])
    prep_1_max = maximum_filter(precipitation, size=3)
    prep_1_max *= ~updraft_ebola_max  # Avoid updraft area
    precipitation_ = precipitation_1_low * prep_1_max
    if np.sum(precipitation) == np.sum(precipitation_):
        break
    precipitation = precipitation_

temp[precipitation] = 20
class8 = temp.copy()

# p = ax[1].pcolormesh(df.data['time'], df.data['range'],
#                      temp.T, cmap=cmap, norm=norm)
# ax[1].yaxis.set_major_formatter(hd.m_km_ticks())
# cbar = fig.colorbar(p, ax=ax[1], ticks=[5, 15, 25, 35, 45])
# cbar.ax.set_yticklabels(['Background', 'Aerosol',
#                          'Precipitation', 'Clouds', 'Undefined'])
# ax[1].set_ylabel('Range [km, a.g.l]')
# ax[1].set_xlabel('Time [UTC - hour]')
# fig.tight_layout()
# fig.savefig(path + '/algorithm_precipitation.png', bbox_inches='tight')

# %%
fig, axes = plt.subplots(2, 2, figsize=(10, 4))
for ax, x in zip(axes.flatten(), [class5, class6, class7, class8]):
    p = ax.pcolormesh(df.data['time'], df.data['range'],
                      x.T,
                      cmap=cmap, norm=norm)
    ax.yaxis.set_major_formatter(hd.m_km_ticks())
    ax.set_ylabel('Height [km]')
    ax.set_xlabel('Time [UTC - hour]')
    cbar = fig.colorbar(p, ax=ax, ticks=[5, 15, 25, 35, 45])
    cbar.ax.set_yticklabels(['Background', 'Aerosol',
                             'Precipitation', 'Clouds', 'Undefined'])
for n, ax in enumerate(axes.flatten()):
    ax.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
            transform=ax.transAxes, size=12)
fig.tight_layout()
fig.savefig('F:/halo/paper/figures/algorithm_precipitation.png', bbox_inches='tight')

# %%
df.data['classifier'][precipitation] = 20

# Remove all aerosol above cloud or precipitation
mask_aerosol0 = df.data['classifier'] == 10
for i in np.array([20, 30]):
    if i == 20:
        mask = df.data['classifier'] == i
    else:
        mask = log_beta > -5
        mask = maximum_filter(mask, size=5)
        mask = median_filter(mask, size=13)
    mask_row = np.argwhere(mask.any(axis=1)).reshape(-1)
    mask_col = np.nanargmax(mask[mask_row, :], axis=1)
    for row, col in zip(mask_row, mask_col):
        mask[row, col:] = True
    mask_undefined = mask * mask_aerosol0
    df.data['classifier'][mask_undefined] = i

# %%

class9 = df.data['classifier'].copy()


# %%
class10 = np.zeros(df.data['beta_raw'].shape, dtype=int)
if (df.data['classifier'] == 10).any():
    classifier = df.data['classifier'].ravel()
    time_dbscan = np.repeat(np.arange(df.data['time'].size),
                            df.data['beta_raw'].shape[1])
    height_dbscan = np.tile(np.arange(df.data['range'].size),
                            df.data['beta_raw'].shape[0])

    time_dbscan = time_dbscan[classifier == 10].reshape(-1, 1)
    height_dbscan = height_dbscan[classifier == 10].reshape(-1, 1)
    X = np.hstack([time_dbscan, height_dbscan])
    db = DBSCAN(eps=3, min_samples=25, n_jobs=-1).fit(X)

    v_dbscan = v_save[classifier == 10]
    range_dbscan = range_save[classifier == 10]

    v_dict = {}
    r_dict = {}
    for i in np.unique(db.labels_):
        v_dict[i] = np.nanmean(v_dbscan[db.labels_ == i])
        r_dict[i] = np.nanmin(range_dbscan[db.labels_ == i])

    lab = db.labels_.copy()
    for key, val in v_dict.items():
        if key == -1:
            lab[db.labels_ == key] = 40
        elif (val < -0.5):
            lab[db.labels_ == key] = 20
        elif r_dict[key] == min(df.data['range']):
            lab[db.labels_ == key] = 10
        elif (val > -0.2):
            lab[db.labels_ == key] = 10
        else:
            lab[db.labels_ == key] = 40

    # df.data['classifier'][df.data['classifier'] == 10] = lab
    temp_ = db.labels_.copy()
    temp_[temp_ == 0] = np.max(db.labels_) + 1
    temp_[temp_ == -1] = np.max(db.labels_) + 2
    class10[df.data['classifier'] == 10] = temp_

# %%
df.data['classifier'][df.data['classifier'] == 10] = lab
class11 = df.data['classifier'].copy()

# %%
# Separate ground rain
if (df.data['classifier'] == 20).any():
    classifier = df.data['classifier'].ravel()
    time_dbscan = np.repeat(np.arange(df.data['time'].size),
                            df.data['beta_raw'].shape[1])
    height_dbscan = np.tile(np.arange(df.data['range'].size),
                            df.data['beta_raw'].shape[0])

    time_dbscan = time_dbscan[classifier == 20].reshape(-1, 1)
    height_dbscan = height_dbscan[classifier == 20].reshape(-1, 1)
    X = np.hstack([time_dbscan, height_dbscan])
    db = DBSCAN(eps=3, min_samples=1, n_jobs=-1).fit(X)

    range_dbscan = range_save[classifier == 20]

    r_dict = {}
    for i in np.unique(db.labels_):
        r_dict[i] = np.nanmin(range_dbscan[db.labels_ == i])

    lab = db.labels_.copy()
    for key, val in r_dict.items():
        if r_dict[key] == min(df.data['range']):
            lab[db.labels_ == key] = 20
        else:
            lab[db.labels_ == key] = 30

    df.data['classifier'][df.data['classifier'] == 20] = lab

# %%
class12 = df.data['classifier'].copy()

# %%
fig, axes = plt.subplots(2, 2, figsize=(10, 4))
for ax, (i, x) in zip(axes.flatten(), enumerate([class9, xxm, class11, class12])):
    if i == 1:
        cmap1 = cm.get_cmap("jet", lut=33)
        cmap1.set_bad("white")
        xxm = np.ma.masked_less(class10, 1)
        p = ax.pcolormesh(df.data['time'], df.data['range'],
                          xxm.T,
                          cmap=cmap1)
        ax.yaxis.set_major_formatter(hd.m_km_ticks())
        cbar = fig.colorbar(p, ax=ax)
        cbar.ax.set_ylabel('Aerosol clusters', rotation=90)
        ax.set_ylabel('Height [km]')
        ax.set_xlabel('Time [UTC - hour]')
    else:
        p = ax.pcolormesh(df.data['time'], df.data['range'],
                          x.T,
                          cmap=cmap, norm=norm)
        ax.yaxis.set_major_formatter(hd.m_km_ticks())
        ax.set_ylabel('Height [km]')
        ax.set_xlabel('Time [UTC - hour]')
        cbar = fig.colorbar(p, ax=ax, ticks=[5, 15, 25, 35, 45])
        cbar.ax.set_yticklabels(['Background', 'Aerosol',
                                 'Precipitation', 'Clouds', 'Undefined'])
for n, ax in enumerate(axes.flatten()):
    ax.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
            transform=ax.transAxes, size=12)
fig.tight_layout()
fig.savefig('F:/halo/paper/figures/algorithm_last.png', bbox_inches='tight')

# %%
####################################
# algorithm examples
####################################
df = xr.open_dataset(r'F:\halo\classifier_new\32/2019-06-26-Uto-32_classified.nc')
data = hd.getdata('F:/halo/32/depolarization/xr')

# %%
date = '20190626'
file = [file for file in data if date in file][0]
df_ = hd.halo_data(file)
df_.filter_height()
df_.unmask999()
df_.depo_cross_adj()
units = {'beta_raw': '$\\log (m^{-1} sr^{-1})$', 'v_raw': '$m s^{-1}$',
         'v_raw_averaged': '$m s^{-1}$',
         'beta_averaged': '$\\log (m^{-1} sr^{-1})$',
         'v_error': '$m s^{-1}$'}
cmap = mpl.colors.ListedColormap(
    ['white', '#2ca02c', 'blue', 'red', 'gray'])
boundaries = [0, 10, 20, 30, 40, 50]
norm = mpl.colors.BoundaryNorm(boundaries, cmap.N, clip=True)
decimal_time = df['time'].dt.hour + \
    df['time'].dt.minute / 60 + df['time'].dt.second/3600
fig, ax = plt.subplots(4, 1, figsize=(6, 8))
ax1, ax3, ax5, ax7 = ax.ravel()
p1 = ax1.pcolormesh(decimal_time, df['range'],
                    np.log10(df_.data['beta_raw']).T, cmap='jet', vmin=-8, vmax=-4)
p2 = ax3.pcolormesh(decimal_time, df['range'],
                    df_.data['v_raw'].T, cmap='jet', vmin=-2, vmax=2)
p3 = ax5.pcolormesh(decimal_time, df['range'],
                    df_.data['co_signal'].T - 1, cmap='jet',
                    vmin=0.995 - 1, vmax=1.005 - 1)
p4 = ax7.pcolormesh(decimal_time, df['range'],
                    df['classified'].T,
                    cmap=cmap, norm=norm)
for ax in [ax1, ax3, ax5, ax7]:
    # ax.yaxis.set_major_formatter(preprocess.m_km_ticks())
    ax.set_ylabel('Height [km]')

cbar = fig.colorbar(p1, ax=ax1)
cbar.ax.set_ylabel(r'$\beta\quad[Mm^{-1}]$', rotation=90)
# cbar.ax.yaxis.set_label_position('left')
cbar = fig.colorbar(p2, ax=ax3)
cbar.ax.set_ylabel('v [' + df_.units.get('v_raw', None) + ']', rotation=90)
# cbar.ax.yaxis.set_label_position('left')
cbar = fig.colorbar(p3, ax=ax5)
cbar.ax.set_ylabel('$SNR_{co}$')
# cbar.ax.yaxis.set_label_position('left')
cbar = fig.colorbar(p4, ax=ax7, ticks=[5, 15, 25, 35, 45])
cbar.ax.set_yticklabels(['Background', 'Aerosol',
                         'Precipitation', 'Clouds', 'Undefined'])
ax7.set_xlabel('Time [UTC - hour]')

fig.tight_layout()
fig.savefig('F:/halo/paper/figures/algorithm_' + df_.filename + '.png', bbox_inches='tight')

# %%
####################################
# algorithm examples
####################################
df = xr.open_dataset(r'F:\halo\classifier_new\53/2019-08-11-Vehmasmaki-53_classified.nc')
data = hd.getdata('F:/halo/53/depolarization/')

# %%
date = '20190811'
file = [file for file in data if date in file][0]
df_ = hd.halo_data(file)
df_.filter_height()
df_.unmask999()
df_.depo_cross_adj()
units = {'beta_raw': '$\\log (m^{-1} sr^{-1})$', 'v_raw': '$m s^{-1}$',
         'v_raw_averaged': '$m s^{-1}$',
         'beta_averaged': '$\\log (m^{-1} sr^{-1})$',
         'v_error': '$m s^{-1}$'}
cmap = mpl.colors.ListedColormap(
    ['white', '#2ca02c', 'blue', 'red', 'gray'])
boundaries = [0, 10, 20, 30, 40, 50]
norm = mpl.colors.BoundaryNorm(boundaries, cmap.N, clip=True)
decimal_time = df['time'].dt.hour + \
    df['time'].dt.minute / 60 + df['time'].dt.second/3600
fig, ax = plt.subplots(4, 1, figsize=(6, 8))
ax1, ax3, ax5, ax7 = ax.ravel()
p1 = ax1.pcolormesh(decimal_time, df['range'],
                    np.log10(df_.data['beta_raw']).T, cmap='jet', vmin=-8, vmax=-4)
p2 = ax3.pcolormesh(decimal_time, df['range'],
                    df_.data['v_raw'].T, cmap='jet', vmin=-2, vmax=2)
p3 = ax5.pcolormesh(decimal_time, df['range'],
                    df_.data['co_signal'].T - 1, cmap='jet',
                    vmin=0.995 - 1, vmax=1.005 - 1)
p4 = ax7.pcolormesh(decimal_time, df['range'],
                    df['classified'].T,
                    cmap=cmap, norm=norm)
for ax in [ax1, ax3, ax5, ax7]:
    # ax.yaxis.set_major_formatter(preprocess.m_km_ticks())
    ax.set_ylabel('Height [km]')

cbar = fig.colorbar(p1, ax=ax1)
cbar.ax.set_ylabel(r'$\beta\quad[Mm^{-1}]$', rotation=90)
# cbar.ax.yaxis.set_label_position('left')
cbar = fig.colorbar(p2, ax=ax3)
cbar.ax.set_ylabel('v [' + df_.units.get('v_raw', None) + ']', rotation=90)
# cbar.ax.yaxis.set_label_position('left')
cbar = fig.colorbar(p3, ax=ax5)
cbar.ax.set_ylabel('$SNR_{co}$')
# cbar.ax.yaxis.set_label_position('left')
cbar = fig.colorbar(p4, ax=ax7, ticks=[5, 15, 25, 35, 45])
cbar.ax.set_yticklabels(['Background', 'Aerosol',
                         'Precipitation', 'Clouds', 'Undefined'])
ax7.set_xlabel('Time [UTC - hour]')

fig.tight_layout()
fig.savefig('F:/halo/paper/figures/algorithm_' + df_.filename + '.png', bbox_inches='tight')

#################################
# %%
#################################

data = hd.getdata('F:/halo/32/depolarization/xr')

# %%
date = '20190405'
file = [file for file in data if date in file][0]
df = hd.halo_data(file)

df.filter_height()
df.unmask999()
df.depo_cross_adj()

# %%
with open('ref_XR2.npy', 'rb') as f:
    ref = np.load(f)
df.filter(variables=['beta_raw'],
          ref='co_signal',
          threshold=1 + 3*df.snr_sd)
# %%
log_beta = np.log10(df.data['beta_raw'])
log_beta2 = log_beta.copy()
log_beta2[:, :50] = log_beta2[:, :50] - ref

# %%
fig, ax = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
for ax_, beta in zip(ax.flatten(), [log_beta, log_beta2]):
    p = ax_.pcolormesh(df.data['time'], df.data['range'],
                       beta.T, cmap='jet', vmin=-8, vmax=-4)
    cbar = fig.colorbar(p, ax=ax_)
    cbar.ax.set_ylabel(r'$\beta\quad[Mm^{-1}]$')
    ax_.yaxis.set_major_formatter(hd.m_km_ticks())
    ax_.set_ylabel('Height [km]')
ax_.set_xlabel('Time [UTC - hour]')
for n, ax_ in enumerate(ax.flatten()):
    ax_.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
             transform=ax_.transAxes, size=12)
fig.tight_layout()
fig.savefig('F:/halo/paper/figures/XR_correction_' + df.filename +
            '.png', bbox_inches='tight')
