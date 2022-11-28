# %% Load modules
from scipy import signal
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import halo_data as hd
from pathlib import Path
import xarray as xr
import matplotlib.dates as dates
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter
from scipy.ndimage import median_filter

#########################################
# %% Hyytiala
#########################################
path = r'F:\halo\paper\figures\case_study\Hyytiala/'

# %%
df = xr.open_dataset(r'F:\halo\classifier_new\46/2018-04-14-Hyytiala-46_classified.nc')
df2 = xr.open_dataset(r'F:\halo\classifier_new\46/2018-04-15-Hyytiala-46_classified.nc')

# %%
delta1 = (df['depo_bleed'].where(df['co_signal'] > 1 + 3 *
          df.attrs['background_snr_sd'])).resample(time='1H').mean()
count1 = (df['depo_bleed'].where(df['co_signal'] > 1 + 3 *
          df.attrs['background_snr_sd'])).resample(time='1H').count()
delta1 = delta1.where(count1 > 20)

delta2 = (df2['depo_bleed'].where(df2['co_signal'] > 1 + 3 *
          df2.attrs['background_snr_sd'])).resample(time='1H').mean()
count2 = (df2['depo_bleed'].where(df2['co_signal'] > 1 + 3 *
          df2.attrs['background_snr_sd'])).resample(time='1H').count()
delta2 = delta2.where(count2 > 20)

beta = np.hstack((df['beta_raw'].T, df2['beta_raw'].T))
# delta = np.hstack((df['depo_bleed'].T, df2['depo_bleed'].T))
delta = np.hstack((delta1.T, delta2.T))
time_plot = np.concatenate((df['time'], df2['time']))
time_delta = np.concatenate((delta1['time'], delta2['time']))

#########################################
# %%


def mean_2d_std(x, window_height=10):
    from scipy import signal
    mask = np.isnan(x)
    mask_ = mask.all(axis=1)
    size2 = x.shape[1]
    x2 = x**2
    top2 = signal.convolve2d(np.where(mask, 0, x2), np.ones((window_height, size2)),
                             mode='full')[:, size2-1]
    top = np.sqrt(top2)
    bottom = signal.convolve2d(~mask, np.ones((window_height, size2)),
                               mode='full')[:, size2-1]
    w1 = int(window_height/2)
    w2 = window_height - 1 - w1
    result = (top/bottom)[w2:-w1]
    result[mask_] = np.nan
    return result


def mean_2d(x, window_height=10):
    from scipy import signal
    mask = np.isnan(x)
    mask_ = mask.all(axis=1)
    size2 = x.shape[1]
    top = signal.convolve2d(np.where(mask, 0, x), np.ones((window_height, size2)),
                            mode='full')[:, size2-1]
    bottom = signal.convolve2d(~mask, np.ones((window_height, size2)),
                               mode='full')[:, size2-1]
    w1 = int(window_height/2)
    w2 = window_height - 1 - w1
    result = (top/bottom)[w2:-w1]
    result[mask_] = np.nan
    return result


def mean_1d_std(x, window_height=10):
    mask = np.isnan(x)
    x2 = x**2
    top2 = np.convolve(np.where(mask, 0, x2), np.ones(window_height),
                       mode='full')
    top = np.sqrt(top2)
    bottom = np.convolve(~mask, np.ones((window_height)),
                         mode='full')
    w1 = int(window_height/2)
    w2 = window_height - 1 - w1
    result = (top/bottom)[w2:-w1]
    result[mask] = np.nan
    return result


def mean_1d(x, window_height=10):
    mask = np.isnan(x)
    top = np.convolve(np.where(mask, 0, x), np.ones(window_height),
                      mode='full')
    bottom = np.convolve(~mask, np.ones(window_height),
                         mode='full')
    w1 = int(window_height/2)
    w2 = window_height - 1 - w1
    result = (top/bottom)[w2:-w1]
    result[mask] = np.nan
    return result


# %%
df_profiles = pd.read_csv(
    r'F:\halo\paper\figures\background_correction_all\stan\46\2018-04-15-Hyytiala-46/2018-04-15-Hyytiala-46_aerosol_bkg_corrected.csv')
df_profiles['time'] = pd.to_datetime(df_profiles['time'])
df_plot = df_profiles[(df_profiles['time'] < '2018-04-15T04:00:00') &
                      (df_profiles['time'] >= '2018-04-15T03:00:00')]
df_plot2 = df_profiles[(df_profiles['time'] < '2018-04-15T08:00:00') &
                       (df_profiles['time'] >= '2018-04-15T06:00:00')]

# %%
time = pd.to_datetime(time_plot)
beta_plot = beta[:, (time < '2018-04-15T04:00:00') & (time >= '2018-04-15T03:00:00')]
beta_plot = mean_2d(beta_plot)

beta_plot2 = beta[:, (time < '2018-04-15T08:00:00') & (time >= '2018-04-15T06:00:00')]
beta_plot2 = mean_2d(beta_plot2)

# %%
df_plot2_wide = df_plot2.pivot(index='range', columns='time')[
    ['depo_corrected', 'depo_corrected_sd']]
df_plot2_range = df_plot2_wide.index.values
df_plot2_wide.columns = df_plot2_wide.columns.droplevel(1)
df_range = pd.DataFrame({'range': np.arange(min(df_plot2['range']), max(df_plot2['range'])+30, 30)})
df_plot2_wide = df_plot2_wide.reset_index().merge(df_range, 'outer')
df_plot2_wide = df_plot2_wide.sort_values('range').set_index('range')

df_range = pd.DataFrame({'range': np.arange(min(df_plot['range']), max(df_plot['range'])+30, 30)})
df_plot_full = df_plot.merge(df_range, 'outer')
df_plot_full = df_plot_full.sort_values('range').set_index('range')
df_plot_range = df_plot['range'].values

# %%
fig, ax = plt.subplots(2, 2, figsize=(9, 4), sharey='row')

c = ax[0, 0].pcolormesh(time_plot, df['range'],
                        beta, norm=LogNorm(vmin=1e-8, vmax=1e-4), cmap='jet')
cbar = fig.colorbar(c, ax=ax[0, 0])

cbar.ax.set_ylabel(r'$\beta\quad[Mm^{-1}]$', rotation=90)

c = ax[0, 1].pcolormesh(time_delta, df['range'],
                        delta, vmin=0, vmax=0.5, cmap='jet')
cbar = fig.colorbar(c, ax=ax[0, 1])

cbar.ax.set_ylabel(r'$\delta$', rotation=90)

hourlocator = dates.HourLocator(byhour=[0, 6, 12, 18, 24])
minorlocator = dates.HourLocator(byhour=np.arange(24))


majorFmt = dates.DateFormatter('%H:%M\n%d-%b')
for ax_ in ax[0]:
    ax_.xaxis.set_major_locator(hourlocator)
    ax_.xaxis.set_minor_locator(minorlocator)
    ax_.xaxis.set_major_formatter(majorFmt)
    ax_.set_xlim(['2018-04-14T18:00:00', '2018-04-15T18:00:00'])
    ax_.set_ylim([0, None])
    ax_.set_xlabel('Time UTC')

    ax_.set_yticks([0, 2000, 4000, 6000, 8000])
    # ax_.axvline(x='2018-04-15T03:00:00', c='gray', ls='--', lw=0.75)
    # ax_.axvline(x='2018-04-15T04:00:00', c='gray', ls='--', lw=0.75)
# ax_.set_xlabel('Time UTC')
ax[0, 0].set_ylabel('Height a.g.l [km]')


mask = np.isin(df['range'], df_plot_range)
mask2 = mean_1d_std(df_plot_full['depo_corrected_sd']) < 0.05
ax[1, 0].scatter(beta_plot[mask], df['range'][mask].values, s=5, alpha=0.7)
ax[1, 0].set_xscale('log')
ax[1, 0].set_xticks([1e-6, 5*1e-7, 1e-7])
ax[1, 1].errorbar(mean_1d(df_plot_full['depo_corrected'])[mask2], df_plot_full.index.values[mask2],
                  xerr=mean_1d_std(df_plot_full['depo_corrected_sd'])[mask2], fmt='.',
                  elinewidth=1, alpha=0.7, ms=3)

ax[1, 0].grid(visible=True, which='both')
ax[1, 1].grid()

ax[1, 0].set_ylabel('Height a.g.l [km]')
ax[1, 0].set_ylim([0, 4000])

ax[1, 0].set_xlabel(r'$\beta$')
ax[1, 1].set_xlabel(r'$\delta$')
for n, ax_ in enumerate(ax.flatten()):
    ax_.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
             transform=ax_.transAxes, size=12)
    ax_.yaxis.set_major_formatter(hd.m_km_ticks())
ax[1, 1].set_xlim([0.05, 0.35])
fig.subplots_adjust(hspace=1)
fig.savefig(path + 'hourly.png', bbox_inches='tight', dpi=500)

#########################################
# %% Uto
#########################################
path = r'F:\halo\paper\figures\case_study\Uto/'

# %%
df = xr.open_dataset(r'F:\halo\classifier_new\32/2017-05-13-Uto-32_classified.nc')
df2 = xr.open_dataset(r'F:\halo\classifier_new\32/2017-05-14-Uto-32_classified.nc')

# %%
delta1 = (df['depo_bleed'].where(df['co_signal'] > 1 + 3 *
          df.attrs['background_snr_sd'])).resample(time='1H').mean()
count1 = (df['depo_bleed'].where(df['co_signal'] > 1 + 3 *
          df.attrs['background_snr_sd'])).resample(time='1H').count()
delta1 = delta1.where(count1 > 20)

delta2 = (df2['depo_bleed'].where(df2['co_signal'] > 1 + 3 *
          df2.attrs['background_snr_sd'])).resample(time='1H').mean()
count2 = (df2['depo_bleed'].where(df2['co_signal'] > 1 + 3 *
          df2.attrs['background_snr_sd'])).resample(time='1H').count()
delta2 = delta2.where(count2 > 20)

beta = np.hstack((df['beta_raw'].T, df2['beta_raw'].T))
# delta = np.hstack((df['depo_bleed'].T, df2['depo_bleed'].T))
delta = np.hstack((delta1.T, delta2.T))
time_plot = np.concatenate((df['time'], df2['time']))
time_delta = np.concatenate((delta1['time'], delta2['time']))

#########################################
# %%
#########################################

# df_profiles = pd.read_csv(r'F:\halo\paper\figures\background_correction_all\stan\32\2017-05-13-Uto-32/2017-05-13-Uto-32_aerosol_bkg_corrected.csv')
df_profiles = pd.concat([pd.read_csv(x) for x in [r'F:\halo\paper\figures\background_correction_all\stan\32\2017-05-13-Uto-32/2017-05-13-Uto-32_aerosol_bkg_corrected.csv',
                                                  r'F:\halo\paper\figures\background_correction_all\stan\32\2017-05-14-Uto-32/2017-05-14-Uto-32_aerosol_bkg_corrected.csv']], ignore_index=True)
df_profiles['time'] = pd.to_datetime(df_profiles['time'])

# %%
df_plot = df_profiles[(df_profiles['time'] >= '2017-05-13T18:00:00') &
                      (df_profiles['time'] < '2017-05-13T20:00:00')]
df_plot = df_plot.drop(df_plot[df_plot['depo_corrected_sd'] > 0.05].index)

df_plot_wide = df_plot.pivot(index='range', columns='time')[
    ['depo_corrected', 'depo_corrected_sd']]
df_plot_range = df_plot_wide.index.values
df_plot_wide.columns = df_plot_wide.columns.droplevel(1)
df_range = pd.DataFrame({'range': np.arange(min(df_plot['range']), max(df_plot['range'])+30, 30)})
df_plot_wide = df_plot_wide.reset_index().merge(df_range, 'outer')
df_plot_wide = df_plot_wide.sort_values('range').set_index('range')


df_plot2 = df_profiles[(df_profiles['time'] >= '2017-05-14T00:00:00') &
                       (df_profiles['time'] < '2017-05-14T05:00:00')]
df_plot2 = df_plot2.drop(df_plot2[df_plot2['depo_corrected_sd'] > 0.05].index)

df_plot2_wide = df_plot2.pivot(index='range', columns='time')[
    ['depo_corrected', 'depo_corrected_sd']]
df_plot2_range = df_plot2_wide.index.values
df_plot2_wide.columns = df_plot2_wide.columns.droplevel(1)
df_range = pd.DataFrame({'range': np.arange(min(df_plot2['range']), max(df_plot2['range'])+30, 30)})
df_plot2_wide = df_plot2_wide.reset_index().merge(df_range, 'outer')
df_plot2_wide = df_plot2_wide.sort_values('range').set_index('range')

# %%
time = pd.to_datetime(time_plot)
beta_plot = beta[:, (time < '2017-05-13T20:00:00') & (time >= '2017-05-13T18:00:00')]
# beta_plot = np.nanmean(beta_plot, axis=1)
beta_plot = mean_2d(beta_plot)

beta_plot2 = beta[:, (time < '2017-05-14T05:00:00') & (time >= '2017-05-14T00:00:00')]
# beta_plot2 = np.nanmean(beta_plot2, axis=1)
beta_plot2 = mean_2d(beta_plot2)

# %%
##########################################
# fig, ax = plt.subplots(3, 2, figsize=(9, 6), sharey='row')
fig = plt.figure(figsize=(9, 6))
ax1 = fig.add_subplot(321)
ax2 = fig.add_subplot(322, sharex=ax1, sharey=ax1)
ax3 = fig.add_subplot(323)
ax4 = fig.add_subplot(324, sharey=ax3)
ax5 = fig.add_subplot(325, sharex=ax3, sharey=ax3)
ax6 = fig.add_subplot(326, sharex=ax4, sharey=ax4)

c = ax1.pcolormesh(time_plot, df['range'],
                   beta, norm=LogNorm(vmin=1e-8, vmax=1e-4), cmap='jet')
cbar = fig.colorbar(c, ax=ax1)

cbar.ax.set_ylabel(r'$\beta\quad[Mm^{-1}]$', rotation=90)

c = ax2.pcolormesh(time_delta, df['range'],
                   delta, vmin=0, vmax=0.5, cmap='jet')
cbar = fig.colorbar(c, ax=ax2)

cbar.ax.set_ylabel(r'$\delta$', rotation=90)

hourlocator = dates.HourLocator(byhour=[0, 6, 12, 18, 24])
minorlocator = dates.HourLocator(byhour=np.arange(24))


majorFmt = dates.DateFormatter('%H:%M\n%d-%b')
for ax_ in [ax1, ax2]:
    ax_.xaxis.set_major_locator(hourlocator)
    ax_.xaxis.set_minor_locator(minorlocator)
    ax_.xaxis.set_major_formatter(majorFmt)
    ax_.set_xlim(['2017-05-13T06:00:00', '2017-05-14T06:00:00'])
    ax_.set_ylim([0, None])
    ax_.set_xlabel('Time UTC')

    ax_.set_yticks([0, 2000, 4000, 6000, 8000])
    # ax_.axvline(x='2018-04-15T03:00:00', c='gray', ls='--', lw=0.75)
    # ax_.axvline(x='2018-04-15T04:00:00', c='gray', ls='--', lw=0.75)
# ax_.set_xlabel('Time UTC')
ax1.set_ylabel('Height a.g.l [km]')


mask = np.isin(df['range'], df_plot_range)
mask2 = mean_2d_std(df_plot_wide['depo_corrected_sd']) < 0.05
ax3.scatter(beta_plot[mask], df['range'][mask].values, s=5, alpha=0.7)
ax3.set_xscale('log')
# ax1.set_xticks([1e-6, 5*1e-7, 1e-7])
ax4.errorbar(mean_2d(df_plot_wide['depo_corrected'])[mask2], df_plot_wide.index.values[mask2],
             xerr=mean_2d_std(df_plot_wide['depo_corrected_sd'])[mask2], fmt='.',
             elinewidth=1, alpha=0.7, ms=3)
# title = group_time.hour
# ax.set_title(str(title).zfill(2) + ':00', weight='bold')
ax3.grid(visible=True, which='both')
ax4.grid()
ax4.set_xlim([0.05, 0.35])
# ax_.xaxis.set_tick_params(labelbottom=True)

# ax2.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
ax3.set_ylabel('Height a.g.l [km]')
ax3.set_ylim([0, 2000])

# ax1.set_xlabel(r'$\beta$')
# ax2.set_xlabel(r'$\delta$')

mask = np.isin(df['range'], df_plot2_range)
mask2 = mean_2d_std(df_plot2_wide['depo_corrected_sd']) < 0.05
ax5.scatter(beta_plot2[mask], df['range'][mask].values, s=5, alpha=0.7)
ax5.set_xscale('log')
# ax3.set_xticks([1e-6, 5*1e-7, 1e-7])
ax6.errorbar(mean_2d(df_plot2_wide['depo_corrected'])[mask2], df_plot2_wide.index.values[mask2],
             xerr=mean_2d_std(df_plot2_wide['depo_corrected_sd'])[mask2], fmt='.',
             elinewidth=1, alpha=0.7, ms=3)
# title = group_time.hour
# ax.set_title(str(title).zfill(2) + ':00', weight='bold')
ax5.grid(visible=True, which='both')
ax6.grid()
ax6.set_xlim([0.05, 0.35])

# ax_.xaxis.set_tick_params(labelbottom=True)

# ax4.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
ax5.set_ylabel('Height a.g.l [km]')
ax5.set_ylim([0, 2000])
# ax5.set_xticks(ax3.get_xticks())

ax5.set_xlabel(r'$\beta$')
ax6.set_xlabel(r'$\delta$')
ax3.set_xlabel(r'$\beta$')
ax4.set_xlabel(r'$\delta$')
for n, ax_ in enumerate(ax.flatten()):
    ax_.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
             transform=ax_.transAxes, size=12)
fig.subplots_adjust(hspace=1)
ax6.set_xlim([0.05, 0.35])

for n, ax_ in enumerate([ax1, ax2, ax3, ax4, ax5, ax6]):
    ax_.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
             transform=ax_.transAxes, size=12)
    ax_.yaxis.set_major_formatter(hd.m_km_ticks())
fig.savefig(path + 'hourly.png', bbox_inches='tight', dpi=500)
