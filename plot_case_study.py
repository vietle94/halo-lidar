# %% Load modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import halo_data as hd
from pathlib import Path
import xarray as xr
import matplotlib.dates as dates
from matplotlib.ticker import FuncFormatter
%matplotlib qt

#########################################
# %% Hyytiala
#########################################
path = r'F:\halo\paper\figures\case_study\Hyytiala/'

# %%
df = xr.open_dataset(r'F:\halo\classifier_new\46/2018-04-14-Hyytiala-46_classified.nc')
df2 = xr.open_dataset(r'F:\halo\classifier_new\46/2018-04-15-Hyytiala-46_classified.nc')

# %%
beta = np.hstack((df['beta_raw'].T, df2['beta_raw'].T))
time = np.concatenate((df['time'], df2['time']))


# %%
fig, ax = plt.subplots(figsize=(7, 2.5))
c = ax.pcolormesh(time, df['range'],
                  np.log10(beta), vmin=-8, vmax=-4, cmap='jet')
cbar = fig.colorbar(c, ax=ax)

cbar.ax.set_ylabel(r'$\beta\quad[Mm^{-1}]$', rotation=90)

hourlocator = dates.HourLocator(byhour=[0, 6, 12, 18, 24])
minorlocator = dates.HourLocator(byhour=np.arange(24))


majorFmt = dates.DateFormatter('%H:%M \n %Y-%m-%d')
# majorFmt = dates.DateFormatter('%H:%M')

ax.xaxis.set_major_locator(hourlocator)
ax.xaxis.set_minor_locator(minorlocator)
ax.xaxis.set_major_formatter(majorFmt)
ax.set_xlabel('Time UTC')
ax.set_xlim(['2018-04-14T18:00:00', '2018-04-16T00:00:00'])
ax.set_ylim([0, None])
ax.set_ylabel('Height a.g.l [km]')
ax.yaxis.set_major_formatter(hd.m_km_ticks())
ax.axvline(x='2018-04-15T00:00:00', c='gray', ls='--', lw=0.75)
ax.axvline(x='2018-04-15T15:00:00', c='gray', ls='--', lw=0.75)

fig.tight_layout()
fig.savefig(path + 'overview.png', bbox_inches='tight')

#########################################
# %%
#########################################

df_profiles = pd.read_csv(
    r'F:\halo\paper\figures\background_correction_all\stan\46\2018-04-15-Hyytiala-46/2018-04-15-Hyytiala-46_aerosol_bkg_corrected.csv')
df_profiles['time'] = pd.to_datetime(df_profiles['time'])
df_plot = df_profiles[df_profiles['time'] < '2018-04-15T15:00:00']

# %%
fig, ax = plt.subplots(3, 5, figsize=(12, 9), sharex=True, sharey=True)
for (group_time, group_value), ax_ in zip(df_plot.groupby(df_plot['time']), ax.flatten()):
    ax_.errorbar(group_value['depo_corrected'], group_value['range'],
                 xerr=group_value['depo_corrected_sd'], fmt='.', elinewidth=1)
    title = group_time.hour
    ax_.set_title(str(title).zfill(2) + ':00', weight='bold')
    ax_.grid()
    # ax_.xaxis.set_tick_params(labelbottom=True)

for ax_ in ax[:, 0]:
    ax_.set_ylabel('Height a.g.l [km]')
    ax_.set_ylim([0, 4000])
    ax_.yaxis.set_major_formatter(hd.m_km_ticks())

for ax_ in ax[-1, :]:
    ax_.set_xlabel(r'$\delta$')
fig.subplots_adjust(hspace=0.3)
fig.savefig(path + 'hourly.png', bbox_inches='tight')


#########################################
# %% Uto
#########################################
path = r'F:\halo\paper\figures\case_study\Uto/'

# %%
df = xr.open_dataset(r'F:\halo\classifier_new\32/2017-05-13-Uto-32_classified.nc')
df2 = xr.open_dataset(r'F:\halo\classifier_new\32/2017-05-14-Uto-32_classified.nc')

# %%
beta = np.hstack((df['beta_raw'].T, df2['beta_raw'].T))
time = np.concatenate((df['time'], df2['time']))


# %%
fig, ax = plt.subplots(figsize=(7, 2.5))
c = ax.pcolormesh(time, df['range'],
                  np.log10(beta), vmin=-8, vmax=-4, cmap='jet')
cbar = fig.colorbar(c, ax=ax)

cbar.ax.set_ylabel(r'$\beta\quad[Mm^{-1}]$', rotation=90)

hourlocator = dates.HourLocator(byhour=[0, 6, 12, 18, 24])
minorlocator = dates.HourLocator(byhour=np.arange(24))


majorFmt = dates.DateFormatter('%H:%M \n %Y-%m-%d')
# majorFmt = dates.DateFormatter('%H:%M')

ax.xaxis.set_major_locator(hourlocator)
ax.xaxis.set_minor_locator(minorlocator)
ax.xaxis.set_major_formatter(majorFmt)
ax.set_xlabel('Time UTC')
ax.set_xlim(['2017-05-13T10:00:00', '2017-05-14T06:00:00'])
ax.set_ylim([0, None])
ax.set_ylabel('Height a.g.l [km]')
ax.yaxis.set_major_formatter(hd.m_km_ticks())
ax.axvline(x='2017-05-13T16:00:00', c='gray', ls='--', lw=0.75)
ax.axvline(x='2017-05-14T04:00:00', c='gray', ls='--', lw=0.75)


fig.tight_layout()
fig.savefig(path + 'overview.png', bbox_inches='tight')

#########################################
# %%
#########################################

# df_profiles = pd.read_csv(r'F:\halo\paper\figures\background_correction_all\stan\32\2017-05-13-Uto-32/2017-05-13-Uto-32_aerosol_bkg_corrected.csv')
df_profiles = pd.concat([pd.read_csv(x) for x in [r'F:\halo\paper\figures\background_correction_all\stan\32\2017-05-13-Uto-32/2017-05-13-Uto-32_aerosol_bkg_corrected.csv',
                                                  r'F:\halo\paper\figures\background_correction_all\stan\32\2017-05-14-Uto-32/2017-05-14-Uto-32_aerosol_bkg_corrected.csv']], ignore_index=True)
df_profiles['time'] = pd.to_datetime(df_profiles['time'])
df_plot = df_profiles[(df_profiles['time'] >= '2017-05-13T16:00:00') &
                      (df_profiles['time'] < '2017-05-14T04:00:00')]
df_plot = df_plot.drop(df_plot[df_plot['depo_corrected_sd'] > 0.05].index)

# %%
fig, ax = plt.subplots(3, 4, figsize=(12/5*4, 9), sharex=True, sharey=True)
for (group_time, group_value), ax_ in zip(df_plot.groupby(df_plot['time']), ax.flatten()):
    ax_.errorbar(group_value['depo_corrected'], group_value['range'],
                 xerr=group_value['depo_corrected_sd'], fmt='.', elinewidth=1)
    title = group_time.hour
    ax_.set_title(str(title).zfill(2) + ':00', weight='bold')
    ax_.grid()
    ax_.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
    ax_.set_xlim([0, 0.4])
    # ax_.xaxis.set_tick_params(labelbottom=True)

for ax_ in ax[:, 0]:
    ax_.set_ylabel('Height a.g.l [km]')
    ax_.set_ylim([0, 2000])
    ax_.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x/1000:.1f}'))

for ax_ in ax[-1, :]:
    ax_.set_xlabel(r'$\delta$')
fig.subplots_adjust(hspace=0.3)
fig.savefig(path + 'hourly.png', bbox_inches='tight')
