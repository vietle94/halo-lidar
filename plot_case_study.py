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
from matplotlib.ticker import FuncFormatter

#########################################
# %% Hyytiala
#########################################
path = r'F:\halo\paper\figures\case_study\Hyytiala/'

# %%
df = xr.open_dataset(r'F:\halo\classifier_new\46/2018-04-14-Hyytiala-46_classified.nc')
df2 = xr.open_dataset(r'F:\halo\classifier_new\46/2018-04-15-Hyytiala-46_classified.nc')

# %%
beta = np.hstack((df['beta_raw'].T, df2['beta_raw'].T))
delta = np.hstack((df['depo_bleed'].T, df2['depo_bleed'].T))
time = np.concatenate((df['time'], df2['time']))


# %%
fig, ax = plt.subplots(1, 2, figsize=(9, 3), sharex=True, sharey=True)
c = ax[0].pcolormesh(time, df['range'],
                     np.log10(beta), vmin=-8, vmax=-4, cmap='jet')
cbar = fig.colorbar(c, ax=ax[0])

cbar.ax.set_ylabel(r'$\beta\quad[Mm^{-1}]$', rotation=90)

c = ax[1].pcolormesh(time, df['range'],
                     delta, vmin=0, vmax=0.5, cmap='jet')
cbar = fig.colorbar(c, ax=ax[1])

cbar.ax.set_ylabel(r'$\delta$', rotation=90)

hourlocator = dates.HourLocator(byhour=[0, 6, 12, 18, 24])
minorlocator = dates.HourLocator(byhour=np.arange(24))


majorFmt = dates.DateFormatter('%H:%M\n%d-%b')
# majorFmt = dates.DateFormatter('%H:%M')
for ax_ in ax:
    ax_.xaxis.set_major_locator(hourlocator)
    ax_.xaxis.set_minor_locator(minorlocator)
    ax_.xaxis.set_major_formatter(majorFmt)
    ax_.set_xlim(['2018-04-14T18:00:00', '2018-04-16T00:00:00'])
    ax_.set_ylim([0, None])
    ax_.set_xlabel('Time UTC')
    ax_.yaxis.set_major_formatter(hd.m_km_ticks())
# ax.axvline(x='2018-04-15T00:00:00', c='gray', ls='--', lw=0.75)
# ax.axvline(x='2018-04-15T15:00:00', c='gray', ls='--', lw=0.75)
# ax_.set_xlabel('Time UTC')
ax[0].set_ylabel('Height a.g.l [km]')
for n, ax_ in enumerate(ax.flatten()):
    ax_.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
             transform=ax_.transAxes, size=12)
fig.tight_layout()
fig.savefig(path + 'overview.png', bbox_inches='tight', dpi=150)

#########################################
# %%
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
time = pd.to_datetime(time)
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
# profile2 = pd.DataFrame({})
# profile2['depo_mean'] = df_plot2.groupby('range')['depo_corrected'
#                                                   ].mean()
# profile2['depo_sd'] = df_plot2.groupby('range')['depo_corrected_sd'
#                                                 ].agg(lambda x: np.sqrt(np.sum(x**2))/len(x))
#
# profile1 = pd.DataFrame({})
# profile1['depo_mean'] = df_plot.groupby('range')['depo_corrected'
#                                                  ].mean()
# profile1['depo_sd'] = df_plot.groupby('range')['depo_corrected_sd'
#                                                ].agg(lambda x: np.sqrt(np.sum(x**2))/len(x))


# %%
# fig, ax = plt.subplots(2, 2, figsize=(9, 4), sharey=True, sharex='col')
# for ax_, profile, beta_ in zip(ax, [profile1, profile2], [beta_plot, beta_plot2]):
#     mask = np.isin(df['range'], profile['range'])
#     mask2 = profile['sum_std'] < 0.05
#     ax_[0].scatter(beta_[mask][mask2], df['range'][mask].values[mask2], s=5, alpha=0.7)
#     ax_[0].set_xscale('log')
#     ax_[0].set_xticks([1e-6, 5*1e-7, 1e-7])
#     ax_[1].errorbar(profile['depo_corrected'][mask2], profile['range'][mask2],
#                     xerr=profile['sum_std'][mask2], fmt='.',
#                     elinewidth=1, alpha=0.7, ms=3)
#     # title = group_time.hour
#     # ax_.set_title(str(title).zfill(2) + ':00', weight='bold')
#     ax_[0].grid(visible=True, which='both')
#     ax_[1].grid()
#     # ax_.xaxis.set_tick_params(labelbottom=True)
#
# for ax_ in ax[:, 0]:
#     ax_.set_ylabel('Height a.g.l [km]')
#     ax_.set_ylim([0, 4000])
#     ax_.yaxis.set_major_formatter(hd.m_km_ticks())
#
# ax[1, 0].set_xlabel(r'$\beta$')
# ax[1, 1].set_xlabel(r'$\delta$')
# for n, ax_ in enumerate(ax.flatten()):
#     ax_.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
#              transform=ax_.transAxes, size=12)
# fig.subplots_adjust(hspace=0.3)
# fig.savefig(path + 'hourly.png', bbox_inches='tight', dpi=150)

# %%
fig, ax = plt.subplots(1, 2, figsize=(9, 2), sharey=True, sharex='col')
mask = np.isin(df['range'], df_plot_range)
mask2 = mean_1d_std(df_plot_full['depo_corrected_sd']) < 0.05
ax[0].scatter(beta_plot[mask], df['range'][mask].values, s=5, alpha=0.7)
ax[0].set_xscale('log')
ax[0].set_xticks([1e-6, 5*1e-7, 1e-7])
ax[1].errorbar(mean_1d(df_plot_full['depo_corrected'])[mask2], df_plot_full.index.values[mask2],
               xerr=mean_1d_std(df_plot_full['depo_corrected_sd'])[mask2], fmt='.',
               elinewidth=1, alpha=0.7, ms=3)
# title = group_time.hour
# ax.set_title(str(title).zfill(2) + ':00', weight='bold')
ax[0].grid(visible=True, which='both')
ax[1].grid()
# ax_.xaxis.set_tick_params(labelbottom=True)

# ax[1].set_xticks([0, 0.1, 0.2, 0.3, 0.4])
ax[0].set_ylabel('Height a.g.l [km]')
ax[0].set_ylim([0, 4000])

ax[0].set_xlabel(r'$\beta$')
ax[1].set_xlabel(r'$\delta$')
for n, ax_ in enumerate(ax.flatten()):
    ax_.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
             transform=ax_.transAxes, size=12)
ax[1].set_xlim([0.05, 0.35])
fig.subplots_adjust(hspace=0.3)
fig.savefig(path + 'hourly.png', bbox_inches='tight', dpi=150)


# %%
# fig, ax = plt.subplots(1, 2, figsize=(9, 2), sharey=True, sharex='col')
# mask = np.isin(df['range'], profile2.index.values)
# mask2 = profile2['depo_sd'] < 0.05
# ax[0].scatter(beta_plot2[mask][mask2], df['range'][mask].values[mask2], s=5, alpha=0.7)
# ax[0].set_xscale('log')
# ax[0].set_xticks([1e-6, 5*1e-7, 1e-7])
# ax[1].errorbar(profile2['depo_mean'][mask2], profile2.index.values[mask2],
#                xerr=profile2['depo_sd'][mask2], fmt='.',
#                elinewidth=1, alpha=0.7, ms=3)
# # title = group_time.hour
# # ax.set_title(str(title).zfill(2) + ':00', weight='bold')
# ax[0].grid(visible=True, which='both')
# ax[1].grid()
# # ax_.xaxis.set_tick_params(labelbottom=True)
#
# # ax[1].set_xticks([0, 0.1, 0.2, 0.3, 0.4])
# ax[0].set_ylabel('Height a.g.l [km]')
# ax[0].set_ylim([0, 4000])
#
# ax[0].set_xlabel(r'$\beta$')
# ax[1].set_xlabel(r'$\delta$')
# for n, ax_ in enumerate(ax.flatten()):
#     ax_.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
#              transform=ax_.transAxes, size=12)
# fig.subplots_adjust(hspace=0.3)
# fig.savefig(path + 'hourly.png', bbox_inches='tight', dpi=150)

#########################################
# %% Uto
#########################################
path = r'F:\halo\paper\figures\case_study\Uto/'

# %%
df = xr.open_dataset(r'F:\halo\classifier_new\32/2017-05-13-Uto-32_classified.nc')
df2 = xr.open_dataset(r'F:\halo\classifier_new\32/2017-05-14-Uto-32_classified.nc')

# %%
beta = np.hstack((df['beta_raw'].T, df2['beta_raw'].T))
delta = np.hstack((df['depo_bleed'].T, df2['depo_bleed'].T))
time = np.concatenate((df['time'], df2['time']))


# %%
fig, ax = plt.subplots(1, 2, figsize=(9, 3), sharex=True, sharey=True)
c = ax[0].pcolormesh(time, df['range'],
                     np.log10(beta), vmin=-8, vmax=-4, cmap='jet')
cbar = fig.colorbar(c, ax=ax[0])

cbar.ax.set_ylabel(r'$\beta\quad[Mm^{-1}]$', rotation=90)

c = ax[1].pcolormesh(time, df['range'],
                     delta, vmin=0, vmax=0.5, cmap='jet')
cbar = fig.colorbar(c, ax=ax[1])

cbar.ax.set_ylabel(r'$\delta$', rotation=90)

hourlocator = dates.HourLocator(byhour=[0, 6, 12, 18, 24])
minorlocator = dates.HourLocator(byhour=np.arange(24))


majorFmt = dates.DateFormatter('%H:%M\n%d-%b')
# majorFmt = dates.DateFormatter('%H:%M')
for ax_ in ax:
    ax_.xaxis.set_major_locator(hourlocator)
    ax_.xaxis.set_minor_locator(minorlocator)
    ax_.xaxis.set_major_formatter(majorFmt)
    ax_.set_xlim(['2017-05-13T00:00:00', '2017-05-14T06:00:00'])
    ax_.set_ylim([0, None])
    ax_.set_xlabel('Time UTC')
    ax_.yaxis.set_major_formatter(hd.m_km_ticks())
# ax.axvline(x='2018-04-15T00:00:00', c='gray', ls='--', lw=0.75)
# ax.axvline(x='2018-04-15T15:00:00', c='gray', ls='--', lw=0.75)
# ax_.set_xlabel('Time UTC')
ax[0].set_ylabel('Height a.g.l [km]')
for n, ax_ in enumerate(ax.flatten()):
    ax_.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
             transform=ax_.transAxes, size=12)
fig.tight_layout()
fig.savefig(path + 'overview.png', bbox_inches='tight', dpi=150)

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
time = pd.to_datetime(time)
beta_plot = beta[:, (time < '2017-05-13T20:00:00') & (time >= '2017-05-13T18:00:00')]
# beta_plot = np.nanmean(beta_plot, axis=1)
beta_plot = mean_2d(beta_plot)

beta_plot2 = beta[:, (time < '2017-05-14T05:00:00') & (time >= '2017-05-14T00:00:00')]
# beta_plot2 = np.nanmean(beta_plot2, axis=1)
beta_plot2 = mean_2d(beta_plot2)

# %%
fig, ax = plt.subplots(1, 2, figsize=(9, 2), sharey=True, sharex='col')
mask = np.isin(df['range'], df_plot_range)
mask2 = mean_2d_std(df_plot_wide['depo_corrected_sd']) < 0.05
ax[0].scatter(beta_plot[mask], df['range'][mask].values, s=5, alpha=0.7)
ax[0].set_xscale('log')
# ax[0].set_xticks([1e-6, 5*1e-7, 1e-7])
ax[1].errorbar(mean_2d(df_plot_wide['depo_corrected'])[mask2], df_plot_wide.index.values[mask2],
               xerr=mean_2d_std(df_plot_wide['depo_corrected_sd'])[mask2], fmt='.',
               elinewidth=1, alpha=0.7, ms=3)
# title = group_time.hour
# ax.set_title(str(title).zfill(2) + ':00', weight='bold')
ax[0].grid(visible=True, which='both')
ax[1].grid()
# ax_.xaxis.set_tick_params(labelbottom=True)

# ax[1].set_xticks([0, 0.1, 0.2, 0.3, 0.4])
ax[0].set_ylabel('Height a.g.l [km]')
ax[0].set_ylim([0, 2000])

ax[0].set_xlabel(r'$\beta$')
ax[1].set_xlabel(r'$\delta$')
for n, ax_ in enumerate(ax.flatten()):
    ax_.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
             transform=ax_.transAxes, size=12)
fig.subplots_adjust(hspace=0.3)
ax[1].set_xlim([0, 0.3])
fig.savefig(path + 'hourly.png', bbox_inches='tight', dpi=150)

# %%
##########################################
fig, ax = plt.subplots(2, 2, figsize=(9, 4), sharey=True, sharex='col')
mask = np.isin(df['range'], df_plot_range)
mask2 = mean_2d_std(df_plot_wide['depo_corrected_sd']) < 0.05
ax[0, 0].scatter(beta_plot[mask], df['range'][mask].values, s=5, alpha=0.7)
ax[0, 0].set_xscale('log')
# ax[0, 0].set_xticks([1e-6, 5*1e-7, 1e-7])
ax[0, 1].errorbar(mean_2d(df_plot_wide['depo_corrected'])[mask2], df_plot_wide.index.values[mask2],
                  xerr=mean_2d_std(df_plot_wide['depo_corrected_sd'])[mask2], fmt='.',
                  elinewidth=1, alpha=0.7, ms=3)
# title = group_time.hour
# ax.set_title(str(title).zfill(2) + ':00', weight='bold')
ax[0, 0].grid(visible=True, which='both')
ax[0, 1].grid()
# ax_.xaxis.set_tick_params(labelbottom=True)

# ax[0, 1].set_xticks([0, 0.1, 0.2, 0.3, 0.4])
ax[0, 0].set_ylabel('Height a.g.l [km]')
ax[0, 0].set_ylim([0, 2000])

# ax[0, 0].set_xlabel(r'$\beta$')
# ax[0, 1].set_xlabel(r'$\delta$')

mask = np.isin(df['range'], df_plot2_range)
mask2 = mean_2d_std(df_plot2_wide['depo_corrected_sd']) < 0.05
ax[1, 0].scatter(beta_plot2[mask], df['range'][mask].values, s=5, alpha=0.7)
ax[1, 0].set_xscale('log')
# ax[1, 0].set_xticks([1e-6, 5*1e-7, 1e-7])
ax[1, 1].errorbar(mean_2d(df_plot2_wide['depo_corrected'])[mask2], df_plot2_wide.index.values[mask2],
                  xerr=mean_2d_std(df_plot2_wide['depo_corrected_sd'])[mask2], fmt='.',
                  elinewidth=1, alpha=0.7, ms=3)
# title = group_time.hour
# ax.set_title(str(title).zfill(2) + ':00', weight='bold')
ax[1, 0].grid(visible=True, which='both')
ax[1, 1].grid()
# ax_.xaxis.set_tick_params(labelbottom=True)

# ax[1, 1].set_xticks([0, 0.1, 0.2, 0.3, 0.4])
ax[1, 0].set_ylabel('Height a.g.l [km]')
ax[1, 0].set_ylim([0, 2000])

ax[1, 0].set_xlabel(r'$\beta$')
ax[1, 1].set_xlabel(r'$\delta$')
for n, ax_ in enumerate(ax.flatten()):
    ax_.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
             transform=ax_.transAxes, size=12)
fig.subplots_adjust(hspace=0.3)
ax[1, 1].set_xlim([0.05, 0.35])
fig.savefig(path + 'hourly.png', bbox_inches='tight', dpi=150)

# %%
# profile = df_plot.groupby('range').mean()
# std1 = df_plot[['range', 'depo_corrected_sd']]
# std1['depo_corrected_sd_2'] = std1['depo_corrected_sd']**2
# profile_std = std1.groupby('range').sum()
# profile['sum_std'] = np.sqrt(profile_std['depo_corrected_sd_2'])/2
#
# profile.reset_index(inplace=True)
#
# # %%
# fig, ax = plt.subplots(1, 2, figsize=(9, 2), sharey=True, sharex='col')
#
# mask = np.isin(df['range'], profile['range'])
# mask2 = profile['sum_std'] < 0.05
# ax[0].scatter(beta_plot[mask][mask2], df['range'][mask].values[mask2], s=5, alpha=0.7)
# ax[0].set_xscale('log')
# # ax[0].set_xticks([1e-6, 1e-7])
# ax[1].errorbar(profile['depo_corrected'][mask2], profile['range'][mask2],
#                xerr=profile['sum_std'][mask2], fmt='.',
#                elinewidth=1, alpha=0.7, ms=3)
# # title = group_time.hour
# # ax.set_title(str(title).zfill(2) + ':00', weight='bold')
# ax[0].grid(visible=True, which='both')
# ax[1].grid()
# # ax_.xaxis.set_tick_params(labelbottom=True)
# ax[1].set_xticks([0, 0.1, 0.2, 0.3, 0.4])
# ax[0].set_ylabel('Height a.g.l [km]')
# ax[0].set_ylim([0, 2000])
# # ax[0].yaxis.set_major_formatter(hd.m_km_ticks())
#
# ax[0].set_xlabel(r'$\beta$')
# ax[1].set_xlabel(r'$\delta$')
# for n, ax_ in enumerate(ax.flatten()):
#     ax_.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
#              transform=ax_.transAxes, size=12)
# fig.subplots_adjust(hspace=0.3)
# fig.savefig(path + 'hourly.png', bbox_inches='tight', dpi=150)
#
# # %%
#
# df_plot = df_profiles[(df_profiles['time'] >= '2017-05-13T16:00:00') &
#                       (df_profiles['time'] < '2017-05-14T04:00:00')]
# df_plot = df_plot.drop(df_plot[df_plot['depo_corrected_sd'] > 0.05].index)
#
# # %%
# fig, ax = plt.subplots(3, 4, figsize=(12/5*4, 9), sharex=True, sharey=True)
# for (group_time, group_value), ax_ in zip(df_plot.groupby(df_plot['time']), ax.flatten()):
#     ax_.errorbar(group_value['depo_corrected'], group_value['range'],
#                  xerr=group_value['depo_corrected_sd'], fmt='.', elinewidth=1)
#     title = group_time.hour
#     ax_.set_title(str(title).zfill(2) + ':00', weight='bold')
#     ax_.grid()
#     ax_.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
#     ax_.set_xlim([0, 0.4])
#     # ax_.xaxis.set_tick_params(labelbottom=True)
#
# for ax_ in ax[:, 0]:
#     ax_.set_ylabel('Height a.g.l [km]')
#     ax_.set_ylim([0, 2000])
#     ax_.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x/1000:.1f}'))
#
# for ax_ in ax[-1, :]:
#     ax_.set_xlabel(r'$\delta$')
# fig.subplots_adjust(hspace=0.3)
# fig.savefig(path + 'hourly.png', bbox_inches='tight')
