import json
from sklearn.metrics import r2_score
import dask.array as da
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
from netCDF4 import Dataset
import pywt
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
std = []
id = []
x_axis = np.linspace(-0.01, 0.1, 100)
for (systemID, group), ax, name in zip(table.groupby('systemID'),
                                       axes.flatten()[1:],
                                       site_names[1:]):
    if systemID == 32:
        x = group['depo'][(group['depo'] > -0.01) & (group['depo'] < 0.1) & (pd.to_datetime(
            group[['year', 'month', 'day']]) < '2017-11-22')]
        axes[0, 0].hist(x, bins=80)

        gmm = GaussianMixture(n_components=2, max_iter=10000)
        gmm.fit(x.values.reshape(-1, 1))
        smean = gmm.means_.ravel()
        sstd = np.sqrt(gmm.covariances_).ravel()
        sort_idx = np.argsort(smean)
        smean = smean[sort_idx]
        sstd = sstd[sort_idx]
        textstr = '\n'.join((
            r'$\mu=%.3f$' % (smean[0], ),
            r'$\sigma=%.3f$' % (sstd[0], )))
        mean.append(smean[0])
        std.append(sstd[0])
        id.append(systemID)
        axes[0, 0].text(0.7, 0.95, textstr, transform=axes[0, 0].transAxes,
                        verticalalignment='top', bbox=props)

        x = group['depo'][(group['depo'] > -0.01) & (group['depo'] < 0.1) & (pd.to_datetime(
            group[['year', 'month', 'day']]) >= '2017-11-22')]
        axes[0, 1].hist(x, bins=80)

        gmm = GaussianMixture(n_components=1, max_iter=10000)
        gmm.fit(x.values.reshape(-1, 1))
        smean = gmm.means_.ravel()
        sstd = np.sqrt(gmm.covariances_).ravel()
        sort_idx = np.argsort(smean)
        smean = smean[sort_idx]
        sstd = sstd[sort_idx]
        textstr = '\n'.join((
            r'$\mu=%.3f$' % (smean[0], ),
            r'$\sigma=%.3f$' % (sstd[0], )))
        mean.append(smean[0])
        std.append(sstd[0])
        id.append(systemID)
        axes[0, 1].text(0.7, 0.95, textstr, transform=axes[0, 1].transAxes,
                        verticalalignment='top', bbox=props)
    elif systemID in [46, 33]:
        x = group['depo'][(group['depo'] > -0.01) & (group['depo'] < 0.1)]
        ax.hist(x, bins=80)

        gmm = GaussianMixture(n_components=2, max_iter=10000)
        gmm.fit(x.values.reshape(-1, 1))
        smean = gmm.means_.ravel()
        sstd = np.sqrt(gmm.covariances_).ravel()
        sort_idx = np.argsort(smean)
        smean = smean[sort_idx]
        sstd = sstd[sort_idx]

        textstr = '\n'.join((
            r'$\mu=%.3f$' % (smean[0], ),
            r'$\sigma=%.3f$' % (sstd[0], )))
        mean.append(smean[0])
        std.append(sstd[0])
        id.append(systemID)
        ax.text(0.7, 0.95, textstr, transform=ax.transAxes,
                verticalalignment='top', bbox=props)
    else:
        x = group['depo'][(group['depo'] > -0.01) & (group['depo'] < 0.1)]
        ax.hist(x, bins=80)
        gmm = GaussianMixture(n_components=1, max_iter=10000)
        gmm.fit(x.values.reshape(-1, 1))
        smean = gmm.means_.ravel()
        sstd = np.sqrt(gmm.covariances_).ravel()
        sort_idx = np.argsort(smean)
        smean = smean[sort_idx]
        sstd = sstd[sort_idx]
        textstr = '\n'.join((
            r'$\mu=%.3f$' % (smean[0], ),
            r'$\sigma=%.3f$' % (sstd[0], )))
        mean.append(smean[0])
        std.append(sstd[0])
        id.append(systemID)
        ax.text(0.7, 0.95, textstr, transform=ax.transAxes,
                verticalalignment='top', bbox=props)
for n, ax in enumerate(axes.flatten()):
    ax.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
            transform=ax.transAxes, size=12)
    ax.set_ylabel('N')
axes.flatten()[-2].set_xlabel('$\delta$')
axes.flatten()[-1].set_xlabel('$\delta$')
fig.subplots_adjust(hspace=0.5, wspace=0.3)
fig.savefig('F:/halo/paper/figures/depo_cloudbase.png', dpi=150,
            bbox_inches='tight')

# %%
# i_plot = 209474
# i_plot = np.random.randint(0, len(table))
# i_plot = np.random.choice(table[table['systemID'] == 46].index.values)
# i_plot = 119216
# i_plot = 235144
# i_plot = 206803
i_plot = 206807
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
mask_range_plot = (df.data['range'] <= cloud_base_height + 150)
mask_time_plot = df.data['time'] == table.iloc[i_plot]['time']
depo_profile_plot = df.data['depo_raw'][mask_time_plot,
                                        mask_range_plot]
co_signal_profile_plot = df.data['co_signal'][mask_time_plot,
                                              mask_range_plot]
beta_profile_plot = df.data['beta_raw'][mask_time_plot,
                                        mask_range_plot]

fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=True)

for ax, var, lab in zip(axes.flatten(), ['depo_raw', 'co_signal', 'beta_raw'],
                        ['$\delta$', '$SNR_{co}$', r'$\beta\quad[Mm^{-1}]$']):
    for h, leg in zip([df.data['range'] <= cloud_base_height,
                       df.data['range'] == cloud_base_height,
                       (cloud_base_height < df.data['range']) &
                       (df.data['range'] <= cloud_base_height + 150)],
                      ['Aerosol', 'Cloud base', 'In-cloud']):
        ax.plot(df.data[var][mask_time_plot, h],
                df.data['range'][h], '.', label=leg)
    ax.grid()
    # ax.axhline(y=cloud_base_height, linestyle='--', color='grey')
    ax.set_xlabel(lab)
axes[2].set_xscale('log')
axes[0].set_xlim([-0.05, 0.1])
# axes[2].set_xlim([1e-8, 1e-3])
axes[0].set_ylabel('Height a.g.l [m]')

for n, ax in enumerate(axes.flatten()):
    ax.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
            transform=ax.transAxes, size=12)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3)
fig.subplots_adjust(bottom=0.2)
print(df.filename)
fig.savefig('F:/halo/paper/figures/' + df.filename + '_depo_profile.png',
            bbox_inches='tight')

# %%
fig1, ax1 = plt.subplots()
ax1.plot(df.data[var][mask_time_plot, df.data['range'] <= cloud_base_height - 150],
         df.data['range'][df.data['range'] <= cloud_base_height - 150], '.', label='xxx')
ax1.plot(df.data[var][mask_time_plot, df.data['range'] == cloud_base_height],
         df.data['range'][df.data['range'] == cloud_base_height], '.', label='yyy')
ax1.plot(df.data[var][mask_time_plot,
                      (df.data['range'] > cloud_base_height) &
                      (df.data['range'] <= cloud_base_height + 150)],
         df.data['range'][(df.data['range'] > cloud_base_height) &
                          (df.data['range'] <= cloud_base_height + 150)], '.',
         label='yyy')
ax1.set_xscale('log')


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

##############################################
# %% depo cloud base hist, ts
##############################################

# Define csv directory path
depo_paths = [
    'F:\\halo\\32\\depolarization\\depo\\xr']

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

# # %%
# device_config_path = glob.glob(r'F:\halo\summary_device_config/*.csv')
# device_config_df = []
# for f in device_config_path:
#     df = pd.read_csv(f)
#     location_sysID = f.split('_')[-1].split('.')[0].split('-')
#     df['location'] = location_sysID[0]
#     df['systemID'] = location_sysID[1]
#     df.rename(columns={'time': 'date'}, inplace=True)
#     df['date'] = pd.to_datetime(df['date'])
#     device_config_df.append(df)
#
# # %%
# device_config = pd.concat(device_config_df, ignore_index=True)
# device_config.loc[(device_config['date'] > '2017-11-20') &
#                   (device_config['systemID'] == '32'),
#                   'systemID'] = '32XR'
# device_config.loc[device_config['systemID'].isin(['32XR', '146']),
#                   'prf'] = 10000
# device_config['integration_time'] = \
#     device_config['num_pulses_m1'] / device_config['prf']
#
# # %%
# result = depo.merge(device_config, on=['systemID', 'date'])
# temp = result.groupby(['systemID', 'date']).mean()

# %%
depo['sys'] = depo['location'] + '-' + depo['systemID']

# %%
fig, ax = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
table32 = table[(table['systemID'] == 32) & (pd.to_datetime(
    table[['year', 'month', 'day']]) >= '2017-11-22')]

co_cross_data = table32[['co_signal', 'cross_signal']].dropna()
H, co_edges, cross_edges = np.histogram2d(
    co_cross_data['co_signal'] - 1,
    co_cross_data['cross_signal'] - 1,
    bins=500)
X, Y = np.meshgrid(co_edges, cross_edges)
p = ax[0].pcolormesh(X, Y, H.T, norm=LogNorm())
ax[0].set_xlabel('$SNR_{co}$')
ax[0].set_ylabel('$SNR_{cross}$')
colorbar = fig.colorbar(p, ax=ax[0])
colorbar.ax.set_ylabel('N')

for key, group in depo.groupby('sys'):
    # Co-cross histogram
    if key == 'Uto-32XR':
        co_cross_data = group[['co_signal', 'cross_signal']].dropna()
        H, co_edges, cross_edges = np.histogram2d(
            co_cross_data['co_signal'] - 1,
            co_cross_data['cross_signal'] - 1,
            bins=1000)
        X, Y = np.meshgrid(co_edges, cross_edges)
        p = ax[1].pcolormesh(X, Y, H.T, norm=LogNorm())
        ax[1].set_xlabel('$SNR_{co}$')
        colorbar = fig.colorbar(p, ax=ax[1])
        colorbar.ax.set_ylabel('N')
        ax[1].set_ylim(top=0.3)
        # ax[0].plot(co_cross_data['co_signal'] - 1,
        #         (co_cross_data['co_signal'] - 1) * 0.01,
        #         label=r'$\frac{cross\_SNR}{co\_SNR} = 0.01$',
        #         linewidth=0.5)
        # ax.legend(loc='upper left')
        # fig5.savefig(path + '/' + key + '_cross_vs_co.png',
        #              bbox_inches='tight')
for n, ax_ in enumerate(ax.flatten()):
    ax_.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
             transform=ax_.transAxes, size=12)
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
        axes[0, 0].set_title('Uto-32')
        # axes[0, 0].plot(co_cross_data['co_signal'] - 1,
        #                 (co_cross_data['co_signal'] - 1) * m + b,
        #                 label=fr'$cross\_SNR = {m:.3f} * co\_SNR {b:.3f}$',
        #                 linewidth=0.5)
        # axes[0, 0].plot(co_cross_data['co_signal'] - 1,
        #                 (co_cross_data['co_signal'] - 1) *
        #                 reg.coef_.flatten()[0] + reg.intercept_.flatten()[0],
        #                 label=fr'$cross\_SNR = {reg.coef_.flatten()[0]:.3f} * co\_SNR$ {reg.intercept_.flatten()[0]:.3f}',
        #                 linewidth=0.5)

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
        axes[0, 1].set_title(name)
        colorbar = fig.colorbar(p, ax=axes[0, 1])
        colorbar.ax.set_ylabel('N')
        # axes[0, 1].plot(co_cross_data['co_signal'] - 1,
        #                 (co_cross_data['co_signal'] - 1) * m + b,
        #                 label=fr'$cross\_SNR = {m:.3f} * co\_SNR {b:.3f}$',
        #                 linewidth=0.5)
        # axes[0, 1].plot(co_cross_data['co_signal'] - 1,
        #                 (co_cross_data['co_signal'] - 1) *
        #                 reg.coef_.flatten()[0] + reg.intercept_.flatten()[0],
        #                 label=fr'$cross\_SNR = {reg.coef_.flatten()[0]:.3f} * co\_SNR$ {reg.intercept_.flatten()[0]:.3f}',
        #                 linewidth=0.5)

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
        ax.set_title(name)
        colorbar = fig.colorbar(p, ax=ax)
        colorbar.ax.set_ylabel('N')
        # ax.plot(co_cross_data['co_signal'] - 1,
        #         (co_cross_data['co_signal'] - 1) * m + b,
        #         label=fr'$cross\_SNR = {m:.3f} * co\_SNR {b:.3f}$',
        #         linewidth=0.5)
        # ax.plot(co_cross_data['co_signal'] - 1,
        #         (co_cross_data['co_signal'] - 1) *
        #         reg.coef_.flatten()[0] + reg.intercept_.flatten()[0],
        #         label=fr'$cross\_SNR = {reg.coef_.flatten()[0]:.3f} * co\_SNR$ {reg.intercept_.flatten()[0]:.3f}',
        #         linewidth=0.5)
for ax in axes.flatten():
    ax.set_xlim([-0.1, 8])
    ax.set_ylim([-0.01, 0.8])
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
date = '20180812'
file = [file for file in data if date in file][0]
df = hd.halo_data(file)

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
# temp_co[lol] = np.nan
temp_co[lol] = 1

fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(8, 2))
p = axes[0].pcolormesh(df.data['time'], df.data['range'],
                       np.log10(df.data['beta_raw']).T, cmap='jet',
                       vmin=-8, vmax=-4)
axes[1].yaxis.set_major_formatter(hd.m_km_ticks())
axes[0].set_yticks([0, 4000, 8000])
axes[0].set_xlim([0, 24])
cbar = fig.colorbar(p, ax=axes[0], fraction=0.05)
cbar.ax.set_ylabel(r'$\beta\quad[Mm^{-1}]$', rotation=90)
cbar.ax.set_title(r'$1e$', size=10)
axes[0].set_ylabel('Height a.g.l [km]')
axes[0].set_xlabel('Time UTC [hour]')

p = axes[1].pcolormesh(df.data['time'], df.data['range'],
                       df.data['v_raw'].T, cmap='jet',
                       vmin=-2, vmax=2)
cbar = fig.colorbar(p, ax=axes[1], fraction=0.05)
cbar.ax.set_ylabel('w [' + df.units.get('v_raw', None) + ']', rotation=90)
# axes[1].set_ylabel('Height a.g.l [km]')
axes[1].set_xlabel('Time UTC [hour]')

p = axes[2].pcolormesh(df.data['time'], df.data['range'],
                       temp_co.T - 1, cmap='jet',
                       vmin=0.995 - 1, vmax=1.005 - 1)
cbar = fig.colorbar(p, ax=axes[2], fraction=0.05)
cbar.ax.set_ylabel('$SNR_{co}$', rotation=90)
# axes[2].set_ylabel('Height a.g.l [km]')
axes[2].set_xlabel('Time UTC [hour]')
axes[2].set_xticks([0, 6, 12, 18, 24])
fig.tight_layout()
fig.savefig('F:/halo/paper/figures/raw_pre-processed.png', dpi=150,
            bbox_inches='tight')

# %%
df.filter_height()
df.unmask999()
df.depo_cross_adj()

# %%
df.filter(variables=['beta_raw', 'v_raw'],
          ref='co_signal',
          threshold=1 + 3*df.snr_sd)
m_ = ((df.data['time'] > 5.70) &
      (df.data['time'] < 5.85)) | ((df.data['time'] > 6.20) &
                                   (df.data['time'] < 6.6))
m_ = m_ | ((df.data['time'] > 17.7) &
           (df.data['time'] < 17.9)) | ((df.data['time'] > 7.02) &
                                        (df.data['time'] < 7.1))

temp_co = df.data['co_signal'].copy()
lol = np.isnan(np.log10(df.data['beta_raw']))
lol[~m_, :] = False
# temp_co[lol] = np.nan
temp_co[lol] = 1

fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(8, 2))
p = axes[0].pcolormesh(df.data['time'], df.data['range'],
                       np.log10(df.data['beta_raw']).T, cmap='jet',
                       vmin=-8, vmax=-4)
axes[1].yaxis.set_major_formatter(hd.m_km_ticks())
axes[0].set_yticks([0, 4000, 8000])
axes[0].set_xlim([0, 24])
cbar = fig.colorbar(p, ax=axes[0], fraction=0.05)
cbar.ax.set_ylabel(r'$\beta\quad[Mm^{-1}]$', rotation=90)
cbar.ax.set_title(r'$1e$', size=10)
axes[0].set_ylabel('Height a.g.l [km]')
axes[0].set_xlabel('Time UTC [hour]')

p = axes[1].pcolormesh(df.data['time'], df.data['range'],
                       df.data['v_raw'].T, cmap='jet',
                       vmin=-2, vmax=2)
cbar = fig.colorbar(p, ax=axes[1], fraction=0.05)
cbar.ax.set_ylabel('w [' + df.units.get('v_raw', None) + ']', rotation=90)
# axes[1].set_ylabel('Height a.g.l [km]')
axes[1].set_xlabel('Time UTC [hour]')

p = axes[2].pcolormesh(df.data['time'], df.data['range'],
                       temp_co.T - 1, cmap='jet',
                       vmin=0.995 - 1, vmax=1.005 - 1)
cbar = fig.colorbar(p, ax=axes[2], fraction=0.05)
cbar.ax.set_ylabel('$SNR_{co}$', rotation=90)
# axes[2].set_ylabel('Height a.g.l [km]')
axes[2].set_xlabel('Time UTC [hour]')
axes[2].set_xticks([0, 6, 12, 18, 24])
axes[0].set_ylim(bottom=0)
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
cmap = mpl.colors.ListedColormap(
    ['white', '#2ca02c', 'blue', 'red', 'gray'])
boundaries = [0, 10, 20, 30, 40, 50]
norm = mpl.colors.BoundaryNorm(boundaries, cmap.N, clip=True)
# fig, ax = plt.subplots(figsize=(8, 2))
# p = ax.pcolormesh(df.data['time'], df.data['range'],
#                   df.data['classifier'].T,
#                   cmap=cmap, norm=norm)
# ax.yaxis.set_major_formatter(hd.m_km_ticks())
# ax.set_ylabel('Height [km, a.g.l]')
# ax.set_xlabel('Time UTC [hour]')
# ax.set_xticks([0, 6, 12, 18, 24])
# ax.set_yticks([0, 4000, 8000])
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
# fig, axes = plt.subplots(2, 2, figsize=(10, 4))
# for ax, x in zip(axes.flatten(), [class1, class2, class3, class4]):
#     p = ax.pcolormesh(df.data['time'], df.data['range'],
#                       x.T,
#                       cmap=cmap, norm=norm)
#     ax.yaxis.set_major_formatter(hd.m_km_ticks())
#     ax.set_ylabel('Height a.g.l [km]')
#     ax.set_xlabel('Time UTC [hour]')
#     ax.set_xticks([0, 6, 12, 18, 24])
#     cbar = fig.colorbar(p, ax=ax, ticks=[5, 15, 25, 35, 45])
#     cbar.ax.set_yticklabels(['Background', 'Aerosol',
#                              'Precipitation', 'Clouds', 'Undefined'])
# for n, ax in enumerate(axes.flatten()):
#     ax.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
#             transform=ax.transAxes, size=12)
# fig.tight_layout()
# fig.savefig('F:/halo/paper/figures/algorithm_cloud_aerosol.png', bbox_inches='tight')
# %%
fig, ax = plt.subplots(figsize=(8, 2))
p = ax.pcolormesh(df.data['time'], df.data['range'],
                  df.data['classifier'].T,
                  cmap=cmap, norm=norm)
ax.yaxis.set_major_formatter(hd.m_km_ticks())
ax.set_ylabel('Height [km, a.g.l]')
ax.set_xlabel('Time UTC [hour]')
ax.set_xticks([0, 6, 12, 18, 24])
ax.set_yticks([0, 4000, 8000])
cbar = fig.colorbar(p, ax=ax, ticks=[5, 15, 25, 35, 45])
cbar.ax.set_yticklabels(['Background', 'Aerosol',
                         'Precipitation', 'Clouds', 'Undefined'])
fig.tight_layout()
fig.savefig('F:/halo/paper/figures/algorithm_aerosol_cloud.png', bbox_inches='tight')


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
        # ax[0].set_ylabel('Height [km, a.g.l]')
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
# ax[1].set_ylabel('Height [km, a.g.l]')
# ax[1].set_xlabel('Time UTC [hour]')
ax.set_xticks([0, 6, 12, 18, 24])
# fig.tight_layout()
# fig.savefig(path + '/algorithm_precipitation.png', bbox_inches='tight')

# %%
fig, ax = plt.subplots(figsize=(8, 2))
p = ax.pcolormesh(df.data['time'], df.data['range'],
                  class8.T,
                  cmap=cmap, norm=norm)
ax.yaxis.set_major_formatter(hd.m_km_ticks())
ax.set_ylabel('Height [km, a.g.l]')
ax.set_xlabel('Time UTC [hour]')
ax.set_xticks([0, 6, 12, 18, 24])
ax.set_yticks([0, 4000, 8000])
cbar = fig.colorbar(p, ax=ax, ticks=[5, 15, 25, 35, 45])
cbar.ax.set_yticklabels(['Background', 'Aerosol',
                         'Precipitation', 'Clouds', 'Undefined'])
fig.tight_layout()
fig.savefig('F:/halo/paper/figures/algorithm_precipitation_detection.png', bbox_inches='tight')

# %%
# fig, axes = plt.subplots(2, 2, figsize=(10, 4))
# for ax, x in zip(axes.flatten(), [class5, class6, class7, class8]):
#     p = ax.pcolormesh(df.data['time'], df.data['range'],
#                       x.T,
#                       cmap=cmap, norm=norm)
#     ax.yaxis.set_major_formatter(hd.m_km_ticks())
#     ax.set_ylabel('Height a.g.l [km]')
#     ax.set_xlabel('Time UTC [hour]')
#     ax.set_xticks([0, 6, 12, 18, 24])
#     cbar = fig.colorbar(p, ax=ax, ticks=[5, 15, 25, 35, 45])
#     cbar.ax.set_yticklabels(['Background', 'Aerosol',
#                              'Precipitation', 'Clouds', 'Undefined'])
# for n, ax in enumerate(axes.flatten()):
#     ax.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
#             transform=ax.transAxes, size=12)
# fig.tight_layout()
# fig.savefig('F:/halo/paper/figures/algorithm_precipitation.png', bbox_inches='tight')

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
fig, ax = plt.subplots(figsize=(8, 2))
p = ax.pcolormesh(df.data['time'], df.data['range'],
                  df.data['classifier'].T,
                  cmap=cmap, norm=norm)
ax.yaxis.set_major_formatter(hd.m_km_ticks())
ax.set_ylabel('Height [km, a.g.l]')
ax.set_xlabel('Time UTC [hour]')
ax.set_xticks([0, 6, 12, 18, 24])
ax.set_yticks([0, 4000, 8000])
cbar = fig.colorbar(p, ax=ax, ticks=[5, 15, 25, 35, 45])
cbar.ax.set_yticklabels(['Background', 'Aerosol',
                         'Precipitation', 'Clouds', 'Undefined'])
fig.tight_layout()
fig.savefig('F:/halo/paper/figures/algorithm_attenuation_correction.png', bbox_inches='tight')

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
fig, ax = plt.subplots(figsize=(8, 2))
p = ax.pcolormesh(df.data['time'], df.data['range'],
                  df.data['classifier'].T,
                  cmap=cmap, norm=norm)
ax.yaxis.set_major_formatter(hd.m_km_ticks())
ax.set_ylabel('Height [km, a.g.l]')
ax.set_xlabel('Time UTC [hour]')
ax.set_xticks([0, 6, 12, 18, 24])
ax.set_yticks([0, 4000, 8000])
cbar = fig.colorbar(p, ax=ax, ticks=[5, 15, 25, 35, 45])
cbar.ax.set_yticklabels(['Background', 'Aerosol',
                         'Precipitation', 'Clouds', 'Undefined'])
fig.tight_layout()
fig.savefig('F:/halo/paper/figures/algorithm_fine-tuned.png', bbox_inches='tight')


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
fig, ax = plt.subplots(figsize=(8, 2))
p = ax.pcolormesh(df.data['time'], df.data['range'],
                  df.data['classifier'].T,
                  cmap=cmap, norm=norm)
ax.yaxis.set_major_formatter(hd.m_km_ticks())
ax.set_ylabel('Height [km, a.g.l]')
ax.set_xlabel('Time UTC [hour]')
ax.set_xticks([0, 6, 12, 18, 24])
ax.set_yticks([0, 4000, 8000])
cbar = fig.colorbar(p, ax=ax, ticks=[5, 15, 25, 35, 45])
cbar.ax.set_yticklabels(['Background', 'Aerosol',
                         'Precipitation', 'Clouds', 'Undefined'])
fig.tight_layout()
fig.savefig('F:/halo/paper/figures/algorithm_ground_precipitation.png', bbox_inches='tight')


# %%
# fig, axes = plt.subplots(2, 2, figsize=(10, 4))
# for ax, (i, x) in zip(axes.flatten(), enumerate([class9, xxm, class11, class12])):
#     if i == 1:
#         cmap1 = cm.get_cmap("jet", lut=33)
#         cmap1.set_bad("white")
#         xxm = np.ma.masked_less(class10, 1)
#         p = ax.pcolormesh(df.data['time'], df.data['range'],
#                           xxm.T,
#                           cmap=cmap1)
#         ax.yaxis.set_major_formatter(hd.m_km_ticks())
#         cbar = fig.colorbar(p, ax=ax)
#         cbar.ax.set_ylabel('Aerosol clusters', rotation=90)
#         ax.set_ylabel('Height a.g.l [km]')
#         ax.set_xlabel('Time UTC [hour]')
#         ax.set_xticks([0, 6, 12, 18, 24])
#     else:
#         p = ax.pcolormesh(df.data['time'], df.data['range'],
#                           x.T,
#                           cmap=cmap, norm=norm)
#         ax.yaxis.set_major_formatter(hd.m_km_ticks())
#         ax.set_ylabel('Height a.g.l [km]')
#         ax.set_xlabel('Time UTC [hour]')
#         ax.set_xticks([0, 6, 12, 18, 24])
#         cbar = fig.colorbar(p, ax=ax, ticks=[5, 15, 25, 35, 45])
#         cbar.ax.set_yticklabels(['Background', 'Aerosol',
#                                  'Precipitation', 'Clouds', 'Undefined'])
# for n, ax in enumerate(axes.flatten()):
#     ax.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
#             transform=ax.transAxes, size=12)
# fig.tight_layout()
# fig.savefig('F:/halo/paper/figures/algorithm_last.png', bbox_inches='tight')

# %%
####################################
# algorithm examples
####################################
df = xr.open_dataset(r'F:\halo\classifier_new\32/2019-06-26-Uto-32_classified.nc')
data = hd.getdata('F:/halo/32/depolarization/xr')

date = '20190626'
file = [file for file in data if date in file][0]
df_ = hd.halo_data(file)
df_.filter_height()
df_.unmask999()
df_.depo_cross_adj()
df_.filter(variables=['beta_raw', 'v_raw'],
           ref='co_signal',
           threshold=1 + 3*df_.snr_sd)
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
p1 = ax1.pcolormesh(decimal_time, df_.data['range'],
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
    ax.set_ylabel('Height a.g.l [km]')
    ax.set_yticks([0, 4000, 8000, 12000])
    ax.yaxis.set_major_formatter(hd.m_km_ticks())
    ax.set_ylim(bottom=0)
    ax.set_xticks([0, 6, 12, 18, 24])

cbar = fig.colorbar(p1, ax=ax1)
cbar.ax.set_ylabel(r'$\beta\quad[Mm^{-1}]$', rotation=90)
cbar.ax.set_title(r'$1e$', size=10)
# cbar.ax.yaxis.set_label_position('left')
cbar = fig.colorbar(p2, ax=ax3)
cbar.ax.set_ylabel('w [' + df_.units.get('v_raw', None) + ']', rotation=90)
# cbar.ax.yaxis.set_label_position('left')
cbar = fig.colorbar(p3, ax=ax5)
cbar.ax.set_ylabel('$SNR_{co}$')
# cbar.ax.yaxis.set_label_position('left')
cbar = fig.colorbar(p4, ax=ax7, ticks=[5, 15, 25, 35, 45])
cbar.ax.set_yticklabels(['Background', 'Aerosol',
                         'Precipitation', 'Clouds', 'Undefined'])
ax7.set_xlabel('Time UTC [hour]')


fig.tight_layout()
fig.savefig('F:/halo/paper/figures/algorithm_' + df_.filename + '.png', bbox_inches='tight')

# %%
####################################
# algorithm examples
####################################
df = xr.open_dataset(r'F:\halo\classifier_new\53/2019-08-11-Vehmasmaki-53_classified.nc')
data = hd.getdata('F:/halo/53/depolarization/')

date = '20190811'
file = [file for file in data if date in file][0]
df_ = hd.halo_data(file)
df_.filter_height()
df_.unmask999()
df_.depo_cross_adj()
df_.filter(variables=['beta_raw', 'v_raw'],
           ref='co_signal',
           threshold=1 + 3*df_.snr_sd)
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
    ax.set_ylabel('Height a.g.l [km]')
    ax.yaxis.set_major_formatter(hd.m_km_ticks())
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_yticks([0, 4000, 8000])
    ax.set_ylim(bottom=0)


cbar = fig.colorbar(p1, ax=ax1)
cbar.ax.set_ylabel(r'$\beta\quad[Mm^{-1}]$', rotation=90)
cbar.ax.set_title(r'$1e$', size=10)
# cbar.ax.yaxis.set_label_position('left')
cbar = fig.colorbar(p2, ax=ax3)
cbar.ax.set_ylabel('w [' + df_.units.get('v_raw', None) + ']', rotation=90)
# cbar.ax.yaxis.set_label_position('left')
cbar = fig.colorbar(p3, ax=ax5)
cbar.ax.set_ylabel('$SNR_{co}$')
# cbar.ax.yaxis.set_label_position('left')
cbar = fig.colorbar(p4, ax=ax7, ticks=[5, 15, 25, 35, 45])
cbar.ax.set_yticklabels(['Background', 'Aerosol',
                         'Precipitation', 'Clouds', 'Undefined'])
ax7.set_xlabel('Time UTC [hour]')

fig.tight_layout()
fig.savefig('F:/halo/paper/figures/algorithm_' + df_.filename + '.png', bbox_inches='tight')

#################################
# %%
#################################

data = hd.getdata('F:/halo/32/depolarization/xr')
date = '20190405'
file = [file for file in data if date in file][0]
df = hd.halo_data(file)

df.filter_height()
df.unmask999()
df.depo_cross_adj()

with open('ref_XR2.npy', 'rb') as f:
    ref = np.load(f)
df.filter(variables=['beta_raw'],
          ref='co_signal',
          threshold=1 + 3*df.snr_sd)

log_beta = np.log10(df.data['beta_raw'])
log_beta2 = log_beta.copy()
log_beta2[:, :50] = log_beta2[:, :50] - ref


fig, ax = plt.subplots(3, 1, figsize=(6, 6), sharex=True)
for ax_, beta in zip(ax.flatten()[:-1], [log_beta, log_beta2]):
    p = ax_.pcolormesh(df.data['time'], df.data['range'],
                       beta.T, cmap='jet', vmin=-8, vmax=-4)
    cbar = fig.colorbar(p, ax=ax_)
    cbar.ax.set_ylabel(r'$\beta\quad[Mm^{-1}]$')
    cbar.ax.set_title(r'$1e$', size=10)
    ax_.yaxis.set_major_formatter(hd.m_km_ticks())
    ax_.set_ylabel('Height a.g.l [km]')
    ax_.set_xticks([0, 6, 12, 18, 24])
    ax_.set_yticks([0, 4000, 8000, 12000])

ceilo = Dataset(r'F:\halo\paper\uto_ceilo_20190405\20190405_uto_cl31.nc')
ax[-1].pcolormesh(ceilo['time'][:], ceilo['range'][:], np.log10(ceilo['beta'][:]).T,
                  vmin=-8, vmax=-4, cmap='jet')
cbar = fig.colorbar(p, ax=ax[-1])
cbar.ax.set_ylabel(r'$\beta\quad[Mm^{-1}]$')
cbar.ax.set_title(r'$1e$', size=10)
ax[-1].set_ylabel('Height a.g.l [km]')
ax[-1].set_xticks([0, 6, 12, 18, 24])
ax[-1].set_yticks([0, 4, 8, 12])
ax[-1].set_xlabel('Time UTC [hour]')

for n, ax_ in enumerate(ax.flatten()):
    ax_.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
             transform=ax_.transAxes, size=12)
fig.tight_layout()
fig.savefig('F:/halo/paper/figures/XR_correction_' + df.filename +
            '.png', bbox_inches='tight')

#######################
# %% depo aerosol statistics
########################
# RH
#################################

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


# %%
plot_data = {'Uto': {}, 'Hyytiala': {},
             'Sodankyla': {}, 'Vehmasmaki': {}}

site_id = {32: 'Uto', 46: 'Hyytiala', 53: 'Vehmasmaki', 54: 'Sodankyla'}
# site_id = {46: 'Hyytiala'}

for site_i, site_x in site_id.items():
    df = xr.open_mfdataset('F:/halo/depo_aerosol/' + str(site_i) + '/*.nc', combine='by_coords')

    threshold = df['aerosol_percentage'] > 0.8
    df_filtered = df.where(threshold)
    depo = (df_filtered.cross_signal_bleed - 1) / (df_filtered.co_signal - 1)
    depo = depo.where(depo < 1)
    depo = depo.where(depo > -0.1)

    # Monthly median
    ##################################
    # month_25 = []
    # month_50 = []
    # month_75 = []
    # month = []
    # for i, x in depo.groupby('time.month'):
    #     x25, x50, x75 = np.nanquantile(x, [0.25, 0.5, 0.75])
    #     month.append(i)
    #     month_25.append(x25)
    #     month_50.append(x50)
    #     month_75.append(x75)
    #
    # plot_data[site_x]['month_25'] = month_25
    # plot_data[site_x]['month_75'] = month_75
    # plot_data[site_x]['month_50'] = month_50
    # plot_data[site_x]['month'] = month

# fig, ax = plt.subplots()
# ax.errorbar(month, month_50, yerr=(np.array(month_50) - np.array(month_25),
#                                    np.array(month_75) - np.array(month_50)))
# ticklabels = [datetime.date(1900, item, 1).strftime('%b') for item
#               in np.arange(1, 13, 3)]
# ax.set_xticks(np.arange(1, 13, 3))
# ax.set_xticklabels(ticklabels)
# ax.set_ylabel('$\delta$')
# fig.savefig('F:/halo/paper/figures/temp/depo_median80.png', bbox_inches='tight')


# Month vs range
#################################
#     monthly_range = depo.groupby('time.month').mean(dim=('time')).load()
# # monthly_range_count = depo.groupby('time.month').count(dim=('time')).load()
# # monthly_range_percentage = monthly_range_count/monthly_range_count.sum(dim='range')
#     plot_data[site_x]['month_range'] = monthly_range
#
# fig, ax = plt.subplots()
# p = ax.pcolormesh(monthly_range.month, monthly_range.range,
#                   monthly_range.T,
#                   vmin=0, vmax=0.3, cmap='jet')
# ax.set_ylim([0, 3000])
# ax.set_xticks(np.arange(1, 13, 3))
# ax.set_xticklabels(ticklabels)
# ax.set_ylabel('Height a.g.l [m]')
# cbar = fig.colorbar(p, ax=ax)
# cbar.ax.set_ylabel('$\delta$')


# Month hour
#####################################
    # month_hour_depo = []
    # month_hour_month = []
    # for month_i, month_x in depo.median(dim='range').groupby('time.month'):
    #     month_hour_depo.append(month_x.groupby('time.hour').median().load().data)
    #     month_hour_month.append(month_i)
    #
    # plot_data[site_x]['month_hour_month'] = month_hour_month
    # plot_data[site_x]['month_hour_depo'] = month_hour_depo
    # plot_data[site_x]['month_hour_x'] = np.arange(24)
    temp = depo.to_dataframe(name='depo')
    temp = temp.reset_index()
    temp['hour'] = temp.time.dt.hour
    temp['month'] = temp.time.dt.month
    temp2 = temp.groupby(['hour', 'month']).median()
    temp2 = temp2.drop('range', axis=1)
    temp3 = temp2.reset_index().pivot(index='month', columns='hour', values='depo')
    plot_data[site_x]['month_hour'] = temp3


# fig, ax = plt.subplots()
# p = ax.pcolormesh(np.arange(24), month_hour_month, np.array(month_hour_depo),
#                   vmin=0, vmax=0.3, cmap='jet')
# cbar = fig.colorbar(p, ax=ax)
# cbar.ax.set_ylabel('$\delta$')
# ax.set_xticks(np.arange(0, 24, 6))
# ax.set_yticks(np.arange(1, 13, 3))
# ax.set_yticklabels(ticklabels)
# ax.set_xlabel('Hour')


# Depo vs range
###################################
    # range_depo_count = []
    # range_depo_range = []
    # for range_i, range_x in depo.where(depo.time.dt.month.isin([5, 6, 7])).groupby('range'):
    #     if range_i > 3000:
    #         break
    #     temp1, depo_edges = da.histogram(range_x.data, bins=300, range=[-0.05, 0.3])
    #     range_depo_count.append(temp1.compute())
    #     range_depo_range.append(range_i)
    #
    # plot_data[site_x]['range_depo_count'] = range_depo_count
    # plot_data[site_x]['range_depo_range'] = range_depo_range
    # plot_data[site_x]['range_depo_edges'] = depo_edges[:-1] + np.diff(depo_edges)/2


# fig, ax = plt.subplots(figsize=(6, 4))
# p = ax.pcolormesh(depo_edges[:-1] + np.diff(depo_edges)/2,
#                   range_depo_range,
#                   np.array(range_depo_count))
# # norm=LogNorm())
# ax.set_xlabel('$\delta$')
# ax.set_ylabel('Height a.g.l [m]')
# ax.set_ylim([0, 3000])
# cbar = fig.colorbar(p, ax=ax)
# cbar.ax.set_ylabel('N')


# Depo vs RH
#############################

    weather_loc = weather[weather['location2'] == site_x]
    weather_loc = weather_loc.set_index('datetime').resample('1H').mean()
    weather_loc = weather_loc.reset_index()
    depo_weather = pd.DataFrame({'datetime': depo.time,
                                 'depo_0': depo[:, 0]})
    weather_ = pd.merge(weather_loc, depo_weather)
    weather_nan = weather_[['datetime', 'RH', 'depo_0']].dropna()

    plot_data[site_x]['weather_nan'] = weather_nan

# fig, ax = plt.subplots()
# ax.plot(weather_nan['RH'], weather_nan['depo_0'], "+",
#         ms=5, mec="k", alpha=0.2)
# z = np.polyfit(weather_nan['RH'], weather_nan['depo_0'], 1)
# y_hat = np.poly1d(z)(weather_nan['RH'])
#
# ax.plot(weather_nan['RH'], y_hat, "r-", lw=1)
# text = f"$y={z[0]:0.3f}\;x{z[1]:+0.3f}$\n$R^2 = {r2_score(weather_nan['depo_0'],y_hat):0.3f}$"
# ax.text(0.05, 0.95, text, transform=ax.transAxes,
#         fontsize=14, verticalalignment='top')
# ax.set_xlabel('Relative humidity [%]')
# ax.set_ylabel('$\delta$')

#####################################
# %% Plot
#########################################
ticklabels = [datetime.date(1900, item, 1).strftime('%b') for item
              in np.arange(1, 13, 3)]
jitter = {}
for place, v in zip(['Uto', 'Hyytiala', 'Vehmasmaki', 'Sodankyla'],
                    np.linspace(-0.15, 0.15, 4)):
    jitter[place] = v
fig, ax = plt.subplots(figsize=(6, 4))
for site in ['Uto', 'Hyytiala', 'Vehmasmaki', 'Sodankyla']:

    ax.errorbar(plot_data[site]['month'] + jitter[site],
                plot_data[site]['month_50'],
                yerr=(np.array(plot_data[site]['month_50']) -
                      np.array(plot_data[site]['month_25']),
                      np.array(plot_data[site]['month_75']) -
                      np.array(plot_data[site]['month_50'])),
                label=site, marker='o',
                fmt='--', elinewidth=1)

    ax.set_xticks(np.arange(1, 13, 3))
    ax.set_xticklabels(ticklabels)
    ax.set_ylabel('$\delta$')
    ax.grid(axis='x', which='major', linewidth=0.5, c='silver')
    ax.grid(axis='y', which='major', linewidth=0.5, c='silver')
    ax.legend()
fig.savefig('F:/halo/paper/figures/monthly_median.png', dpi=150,
            bbox_inches='tight')

# %%
fig, axes = plt.subplots(2, 2, figsize=(6, 4), sharey=True, sharex=True)
for site, ax in zip(['Uto', 'Hyytiala', 'Vehmasmaki', 'Sodankyla'],
                    axes.flatten()):
    p = ax.pcolormesh(plot_data[site]['month_range'].month,
                      plot_data[site]['month_range'].range,
                      plot_data[site]['month_range'].T,
                      vmin=0, vmax=0.3, cmap='jet')
    ax.set_ylim([0, 3000])
    ax.set_xticks(np.arange(1, 13, 3))
    cbar = fig.colorbar(p, ax=ax)
    cbar.ax.set_ylabel('$\delta$')
for ax in [axes[0, 0], axes[1, 0]]:
    ax.set_ylabel('Height a.g.l [km]')
    ax.yaxis.set_major_formatter(hd.m_km_ticks())
for ax in [axes[1, 0], axes[1, 1]]:
    ax.set_xticklabels(ticklabels)
for n, ax in enumerate(axes.flatten()):
    ax.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
            transform=ax.transAxes, size=12)
fig.savefig('F:/halo/paper/figures/monthly_range_depo.png', dpi=150,
            bbox_inches='tight')

# %%
fig, axes = plt.subplots(2, 2, figsize=(6, 4), sharey=True, sharex=True)
for site, ax in zip(['Uto', 'Hyytiala', 'Vehmasmaki', 'Sodankyla'],
                    axes.flatten()):
    p = ax.pcolormesh(np.arange(24), np.arange(1, 13),
                      plot_data[site]['month_hour'],
                      vmin=0, vmax=0.3, cmap='jet')
    cbar = fig.colorbar(p, ax=ax)
    cbar.ax.set_ylabel('$\delta$')
    ax.set_xticks(np.arange(0, 24, 6))
    ax.set_yticks(np.arange(1, 13, 3))

for ax in [axes[0, 0], axes[1, 0]]:
    ax.set_yticklabels(ticklabels)
for ax in [axes[1, 0], axes[1, 1]]:
    ax.set_xlabel('Hour')
for n, ax in enumerate(axes.flatten()):
    ax.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
            transform=ax.transAxes, size=12)
fig.savefig('F:/halo/paper/figures/monthly_hour_depo.png', dpi=150,
            bbox_inches='tight')

# %%
fig, axes = plt.subplots(2, 2, figsize=(6, 4), sharey=True, sharex=True)
for site, ax in zip(['Uto', 'Hyytiala', 'Vehmasmaki', 'Sodankyla'],
                    axes.flatten()):
    h = np.array(plot_data[site]['range_depo_count']).astype(float)
    h[h == 0] = np.nan
    p = ax.pcolormesh(plot_data[site]['range_depo_edges'],
                      plot_data[site]['range_depo_range'],
                      np.array(plot_data[site]['range_depo_count']))
    # norm=LogNorm())
    ax.set_ylim([0, 3000])
    cbar = fig.colorbar(p, ax=ax)
    cbar.ax.set_ylabel('N')
for ax in [axes[0, 0], axes[1, 0]]:
    ax.set_ylabel('Height a.g.l [m]')
for ax in [axes[1, 0], axes[1, 1]]:
    ax.set_xlabel('$\delta$')
for n, ax in enumerate(axes.flatten()):
    ax.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
            transform=ax.transAxes, size=12)
fig.savefig('F:/halo/paper/figures/depo_range_summer.png', dpi=150,
            bbox_inches='tight')

# %%
fig, axes = plt.subplots(2, 2, figsize=(6, 4), sharey=True, sharex=True)
for site, ax in zip(['Uto', 'Hyytiala', 'Vehmasmaki', 'Sodankyla'],
                    axes.flatten()):
    ax.plot(plot_data[site]['weather_nan']['RH'],
            plot_data[site]['weather_nan']['depo_0'], "+",
            ms=5, mec="k", alpha=0.05)
    z = np.polyfit(plot_data[site]['weather_nan']['RH'],
                   plot_data[site]['weather_nan']['depo_0'], 1)
    y_hat = np.poly1d(z)(plot_data[site]['weather_nan']['RH'])

    ax.plot(plot_data[site]['weather_nan']['RH'],
            y_hat, "r-", lw=1)
    text = f"$y={z[0]:0.3f}\;x{z[1]:+0.3f}$\n$R^2 = {r2_score(plot_data[site]['weather_nan']['depo_0'],y_hat):0.3f}$"
    ax.text(0.05, 0.95, text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top')

for ax in [axes[0, 0], axes[1, 0]]:
    ax.set_ylabel('$\delta$')
for ax in [axes[1, 0], axes[1, 1]]:
    ax.set_xlabel('Relative humidity [%]')
for n, ax in enumerate(axes.flatten()):
    ax.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
            transform=ax.transAxes, size=12)
fig.savefig('F:/halo/paper/figures/RH_depo_point.png', dpi=150,
            bbox_inches='tight')

# %%
fig, axes = plt.subplots(2, 2, figsize=(6, 4), sharey=True, sharex=True)
for site, ax in zip(['Uto', 'Hyytiala', 'Vehmasmaki', 'Sodankyla'],
                    axes.flatten()):
    RH_depo_data = plot_data[site]['weather_nan'][['RH', 'depo_0']].dropna()
    H, RH_edges, depo_0_edges = np.histogram2d(
        RH_depo_data['RH'],
        RH_depo_data['depo_0'],
        bins=100)
    z = np.polyfit(plot_data[site]['weather_nan']['RH'],
                   plot_data[site]['weather_nan']['depo_0'], 1)
    y_hat = np.poly1d(z)(plot_data[site]['weather_nan']['RH'])

    ax.plot(plot_data[site]['weather_nan']['RH'],
            y_hat, "r-", lw=1)
    text = f"$y={z[0]:0.3f}\;x{z[1]:+0.3f}$\n$R^2 = {r2_score(plot_data[site]['weather_nan']['depo_0'],y_hat):0.3f}$"
    ax.text(0.05, 0.95, text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top')
    X, Y = np.meshgrid(RH_edges, depo_0_edges)
    H[H == 0] = np.nan
    p = ax.pcolormesh(X, Y, H.T)
    cbar = fig.colorbar(p, ax=ax)
    cbar.ax.set_ylabel('N')

for ax in [axes[0, 0], axes[1, 0]]:
    ax.set_ylabel('$\delta$')
for ax in [axes[1, 0], axes[1, 1]]:
    ax.set_xlabel('Relative humidity [%]')
for n, ax in enumerate(axes.flatten()):
    ax.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
            transform=ax.transAxes, size=12)
fig.savefig('F:/halo/paper/figures/RH_depo_2d.png', dpi=150,
            bbox_inches='tight')

# %%
period_months = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
for site in ['Uto', 'Hyytiala', 'Vehmasmaki', 'Sodankyla']:
    fig, axes = plt.subplots(2, 2, figsize=(6, 4), sharey=True, sharex=True)
    for period, ax in zip(period_months, axes.flatten()):
        temp = plot_data[site]['weather_nan'][
            plot_data[site]['weather_nan'].datetime.dt.month.isin(period)]
        ax.plot(temp['RH'],
                temp['depo_0'], "+",
                ms=5, mec="k", alpha=0.05)
        z = np.polyfit(temp['RH'],
                       temp['depo_0'], 1)
        y_hat = np.poly1d(z)(temp['RH'])

        ax.plot(temp['RH'],
                y_hat, "r-", lw=1)
        text = f"$y={z[0]:0.3f}\;x{z[1]:+0.3f}$\n$R^2 = {r2_score(temp['depo_0'],y_hat):0.3f}$"
        ax.text(0.05, 0.95, text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top')
    fig.savefig('F:/halo/paper/figures/RH_depo_point' + site,
                dpi=150, bbox_inches='tight')

#####################################
# %% case study
#########################################


def bleed_through(df):
    # Correction for bleed through and remove all observations below 90m
    with open('summary_info.json', 'r') as file:
        summary_info = json.load(file)
    df = df.where(df.range > 90, drop=True)
    file_date = '-'.join([str(int(df.attrs[ele])).zfill(2) for
                          ele in ['year', 'month', 'day']])
    file_location = '-'.join([df.attrs['location'], str(int(df.attrs['systemID']))])
    df.attrs['file_name'] = file_date + '-' + file_location

    if '32' in file_location:
        for period in summary_info['32']:
            if (period['start_date'] <= file_date) & \
                    (file_date <= period['end_date']):
                df.attrs['background_snr_sd'] = period['snr_sd']
                df.attrs['bleed_through_mean'] = period['bleed_through']['mean']
                df.attrs['bleed_through_sd'] = period['bleed_through']['sd']
    else:
        id = str(int(df.attrs['systemID']))
        df.attrs['background_snr_sd'] = summary_info[id]['snr_sd']
        df.attrs['bleed_through_mean'] = summary_info[id]['bleed_through']['mean']
        df.attrs['bleed_through_sd'] = summary_info[id]['bleed_through']['sd']
    bleed = df.attrs['bleed_through_mean']
    sigma_bleed = df.attrs['bleed_through_sd']
    sigma_co, sigma_cross = df.attrs['background_snr_sd'], df.attrs['background_snr_sd']

    df['cross_signal_bleed'] = (['time', 'range'], ((df['cross_signal'] - 1) -
                                                    bleed * (df['co_signal'] - 1) + 1).data)

    df['cross_signal_bleed_sd'] = np.sqrt(
        sigma_cross**2 +
        ((bleed * (df['co_signal'] - 1))**2 *
         ((sigma_bleed/bleed)**2 +
          (sigma_co/(df['co_signal'] - 1))**2))
    )
    df['depo_bleed'] = (df['cross_signal_bleed'] - 1) / \
        (df['co_signal'] - 1)

    df['depo_bleed_sd'] = np.sqrt(
        (df['depo_bleed'])**2 *
        (
            (df['cross_signal_bleed_sd']/(df['cross_signal_bleed'] - 1))**2 +
            (sigma_co/(df['co_signal']-1))**2
        ))
    return df


# %%
df = xr.open_dataset(r'F:\halo\classifier_new\46/2018-04-15-Hyytiala-46_classified.nc')
df = bleed_through(df)

# %%
avg = df[['co_signal', 'cross_signal_bleed']].resample(time='60min').mean(dim='time')

co = avg['co_signal'][0, :].values - 1
cross = avg['cross_signal_bleed'][0, :].values - 1

# %%


def wavelet_denoising(x, wavelet='db4', level=1):
    coeff = pywt.wavedec(x, wavelet, mode="symmetric")
    sigma = (1/0.6745) * madev(coeff[-level])
    # uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    uthresh = np.median(np.abs(coeff[1]))/0.6745 * np.sqrt(2 * np.log(len(coeff[1])))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode='symmetric')

# %%


# %%
fig, ax = plt.subplots(1, 6, sharey=True, sharex=True)
for i in range(6):
    ax[i].plot(avg['co_signal'][i, :].values, df['range'])
    ax[i].plot(avg['cross_signal_bleed'][0, :].values, df['range'])

# %%
time_mask = df['time'].dt.hour < 1
co = np.nanmean(df['co_signal'][time_mask, :], axis=0)
cross = np.nanmean(df['cross_signal'][time_mask, :], axis=0)
classified = np.sum(df['classified'][time_mask, :], axis=0)
n = df['co_signal'][time_mask, :].shape[0]
selected_range = df['range'][co < 1 + 3*df.attrs['background_snr_sd']/np.sqrt(n)]
selected_co = co[co < 1 + 3*df.attrs['background_snr_sd']/np.sqrt(n)]
selected_cross = cross[co < 1 + 3*df.attrs['background_snr_sd']/np.sqrt(n)]

# %%
a, b, c = np.polyfit(selected_range, selected_co, deg=2)
y_co = c + b*df['range'] + a*(df['range']**2)
y_co_background = c + b*selected_range + a*(selected_range**2)

a, b, c = np.polyfit(selected_range, selected_cross, deg=2)
y_cross = c + b*df['range'] + a*(df['range']**2)
y_cross_background = c + b*selected_range + a*(selected_range**2)

# %%
fig, axes = plt.subplots(1, 3, sharex=True, sharey=True,
                         figsize=(9, 4))

axes[0].plot(co, df['range'], '.', label='$SNR_{co}$')
axes[0].plot(cross, df['range'], '.', label='$SNR_{cross}$')

axes[2].plot(selected_co, selected_range, '+', label='$SNR_{co}$')
axes[2].plot(selected_cross, selected_range, '+', label='$SNR_{cross}$')

axes[0].plot(y_co, df['range'], label='Fitted $SNR_{co}$')
axes[0].plot(y_cross, df['range'], label='Fitted $SNR_{cross}$')

axes[1].plot(co/y_co, df['range'], '.', label='Corrected $SNR_{co}$')
axes[1].plot(cross/y_cross, df['range'], '.', label='Corrected $SNR_{cross}$')

axes[0].set_xlim([0.9995, 1.001])
axes[0].yaxis.set_major_formatter(hd.m_km_ticks())
axes[0].set_ylabel('Height a.g.l [km]')
for ax in axes.flatten():
    ax.tick_params(axis='x', labelrotation=45)
    ax.legend()
    ax.set_xlabel('SNR')
fig.savefig('temp.png', bbox_inches='tight')

# %%
filter_co = df.co_signal > 1 + 3 * df.attrs['background_snr_sd']
df['cross_signal_bleed'] = df.cross_signal_bleed.where(filter_co)
df['classified'] = df.classified.where(filter_co)
df['co_signal'] = df.co_signal.where(filter_co)

df['co_bg_corrected'] = df['co_signal']/y_co
df['cross_bg_corrected'] = df['cross_signal_bleed']/y_cross
filter_aerosol = df.classified == 10
hour1 = df[['co_bg_corrected', 'cross_bg_corrected', 'co_signal', 'cross_signal_bleed']].where(
    filter_aerosol).resample(time='60min').mean()
hour1['aerosol_percentage'] = filter_aerosol.resample(time='60min').mean()

hour1_80 = hour1.where(hour1['aerosol_percentage'] > 0.8)
hour1_boundary_aerosol = hour1_80.where((hour1_80.time.dt.hour < 12) & (hour1_80.range < 2000))

depo_corrected = np.array(((hour1_boundary_aerosol['cross_bg_corrected'] - 1) /
                           (hour1_boundary_aerosol['co_bg_corrected'] - 1)).load())
depo = np.array(((hour1_boundary_aerosol['cross_signal_bleed'] - 1) /
                 (hour1_boundary_aerosol['co_signal'] - 1)).load())

depo_corrected = depo_corrected[~np.isnan(depo_corrected)]
depo = depo[~np.isnan(depo)]

# %%
fig, ax = plt.subplots(figsize=(6, 4), sharex=True, sharey=True)
ax.hist(depo_corrected, label='Corrected depo')
ax.hist(depo, alpha=0.5, label='Not corrected depo')
ax.legend()
ax.set_xlabel('$\delta$')
ax.set_ylabel('N')
fig.savefig('temp.png', bbox_inches='tight')

# %%


def gaussian(x, s):
    return 1./np.sqrt(2. * np.pi * s**2) * np.exp(-x**2 / (2. * s**2))


# %%
gaus = np.array([gaussian(x, 1) for x in range(-13, 14, 1)])

# %%
coef = pywt.wavedec(np.convolve(co, gaus, 'same'), 'haar', level=1)
fig, ax = plt.subplots(len(coef) + 1, 1, figsize=(9, 6))
ax[0].plot(co)
ax[1].plot(pywt.idwt(coef[0], None, 'haar'))
for i in range(1, len(coef)):
    ax[i+1].plot(pywt.idwt(None, coef[i], 'haar'))


# %%


def lowpassfilter(signal, thresh=0.63, wavelet="haar"):
    thresh = thresh*np.nanmax(signal)
    coeff = pywt.wavedec(signal, wavelet, mode="per")
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft") for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per")
    return reconstructed_signal

# def lowpassfilter(signal, thresh = 0.63, wavelet="haar"):
#     thresh = thresh*np.nanstd(signal)
#     coeff = pywt.wavedec(signal, wavelet, mode="per" )
#     coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft" ) for i in coeff[1:])
#     reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per" )
#     return reconstructed_signal


fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(co-1, color="b", alpha=0.5, label='original signal')
rec = lowpassfilter(co-1, 0.1)
ax.plot(rec, 'k', label='DWT smoothing}', linewidth=2)
ax.legend()
ax.set_title('Removing High Frequency Noise with DWT', fontsize=18)
ax.set_ylabel('Signal Amplitude', fontsize=16)
ax.set_xlabel('Sample No', fontsize=16)
plt.show()

# %%
fig, ax = plt.subplots()
ax.plot(df['range'], co-1)
ax.plot(df['range'], np.convolve(co-1, [0, 1, -1, 0], 'same'))

# %%


def gaussian(x, s):
    return 1./np.sqrt(2. * np.pi * s**2) * np.exp(-x**2 / (2. * s**2))


# %%
gaus = np.array([gaussian(x, 1) for x in range(-3, 4, 1)])
fig, ax = plt.subplots()
ax.plot(df['range'], co-1)
ax.plot(df['range'], np.convolve(co-1, gaus, 'same'))
ax.plot(df['range'], np.convolve(np.convolve(co-1, gaus, 'same'), [0, 1, -1, 0], 'same'))


# %%
plt.plot(np.convolve(co-1, [0, 1, -1, 0]))
plt.plot(co-1)

# %%
coef = pywt.wavedec(co, 'sym6', level=1)
threshold = np.median(np.abs(coef[1]))/0.6745 * np.sqrt(2 * len(coef[1]))
coef[1:] = (pywt.threshold(i, value=threshold, mode="soft") for i in coef[1:])
reconstructed_signal = pywt.waverec(coef, 'sym6')

# %%
plt.plot(reconstructed_signal)
plt.plot(co)
reconstructed_signal.size
co.size

# %%


def madev(d, axis=None):
    """ Mean absolute deviation of a signal """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def wavelet_denoising(x, wavelet='db4', level=1):
    coeff = pywt.wavedec(x, wavelet, mode="symmetric")
    sigma = (1/0.6745) * madev(coeff[-level])
    # uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    uthresh = np.median(np.abs(coeff[1]))/0.6745 * np.sqrt(2 * np.log(len(coeff[1])))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode='symmetric')


# %%
# for wav in pywt.wavelist():
for wav in ['sym5', 'db7', 'db5', 'bior4.4']:
    try:
        filtered = wavelet_denoising(co, wavelet=wav, level=1)
    except:
        pass

    plt.figure(figsize=(10, 6))
    plt.plot(co, label='Raw')
    plt.plot(filtered, label='Filtered')
    plt.axhline(y=0)

    plt.ylim([-0.0002, 0.0003])
    plt.legend()
    plt.title(f"DWT Denoising with {wav} Wavelet", size=15)
    plt.show()

# %%

filtered = wavelet_denoising(co, wavelet='bior4.4', level=1)
plt.figure(figsize=(10, 6))
plt.plot(co, label='Raw')
plt.plot(filtered, label='Filtered')
plt.axhline(y=6e-5)

plt.ylim([-0.0002, 0.0003])
plt.legend()
plt.title(f"DWT Denoising with {wav} Wavelet", size=15)
plt.show()
