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
    axes[1].plot(df['time'], df['noise'] * np.sqrt(df['integration_time']),
                 '.', label=site,
                 markeredgewidth=0.0)
    axes[1].set_ylabel('$\sigma$ x $\sqrt{integration\_time}$')
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
####################################
# Depo cloudbase
####################################
site_names = ['Uto-32', 'Uto-32XR', 'Hyytiala-33',
              'Hyytiala-46', 'Vehmasmaki-53', 'Sodankyla-54']
fig, axes = plt.subplots(3, 2, figsize=(9, 6), sharex=True)
table = pd.read_csv('F:/halo/paper/depo_cloudbase/result.csv')
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
for (systemID, group), ax, name in zip(table.groupby('systemID'),
                                       axes.flatten()[1:],
                                       site_names[1:]):
    if systemID == 32:
        x = group['depo'][(group['depo'] > -0.01) & (group['depo'] < 0.1) & (pd.to_datetime(
            group[['year', 'month', 'day']]) < '2017-11-22')]
        axes[0, 0].hist(x, bins=20)
        textstr = '\n'.join((
            r'$\mu=%.3f$' % (np.mean(x), ),
            r'$\mathrm{median}=%.3f$' % (np.median(x), ),
            r'$\sigma=%.3f$' % (np.std(x), )))

        axes[0, 0].text(0.6, 0.95, textstr, transform=axes[0, 0].transAxes,
                        verticalalignment='top', bbox=props)

        x = group['depo'][(group['depo'] > -0.01) & (group['depo'] < 0.1) & (pd.to_datetime(
            group[['year', 'month', 'day']]) >= '2017-11-22')]
        axes[0, 1].hist(x, bins=20)
        textstr = '\n'.join((
            r'$\mu=%.3f$' % (np.mean(x), ),
            r'$\mathrm{median}=%.3f$' % (np.median(x), ),
            r'$\sigma=%.3f$' % (np.std(x), )))

        axes[0, 1].text(0.6, 0.95, textstr, transform=axes[0, 1].transAxes,
                        verticalalignment='top', bbox=props)
        # axes[0, 1].set_title('Uto-32XR', weight='bold')
    else:
        x = group['depo'][(group['depo'] > -0.01) & (group['depo'] < 0.1)]
        ax.hist(x, bins=20)
        textstr = '\n'.join((
            r'$\mu=%.3f$' % (np.mean(x), ),
            r'$\mathrm{median}=%.3f$' % (np.median(x), ),
            r'$\sigma=%.3f$' % (np.std(x), )))

        ax.text(0.6, 0.95, textstr, transform=ax.transAxes,
                verticalalignment='top', bbox=props)
        # ax.set_title(name, weight='bold')
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
i_plot = np.random.randint(0, len(table))
i_plot = 119216
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
mask_range_plot = (df.data['range'] <= table.iloc[i_plot]['range'] + 300)
mask_time_plot = df.data['time'] == table.iloc[i_plot]['time']
depo_profile_plot = df.data['depo_raw'][mask_time_plot,
                                        mask_range_plot]
co_signal_profile_plot = df.data['co_signal'][mask_time_plot,
                                              mask_range_plot]
beta_profile_plot = df.data['beta_raw'][mask_time_plot,
                                        mask_range_plot]

fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
axes[0].plot(depo_profile_plot, df.data['range'][mask_range_plot], '.')
axes[0].axhline(y=table.iloc[i_plot]['range'], linestyle='--')
axes[0].set_xlim([-0.05, 0.1])
axes[0].set_xlabel('$\delta$')
axes[0].set_ylabel('Height [m]')

axes[1].plot(co_signal_profile_plot - 1,
             df.data['range'][mask_range_plot], '.')
axes[1].axhline(y=table.iloc[i_plot]['range'], linestyle='--')
# axes[1].set_xlim([-0.1, 0.5])
axes[1].set_xlabel('$SNR_{co}$')
# axes[1].set_ylabel('Height (km)')

axes[2].plot(np.log10(beta_profile_plot), df.data['range'][mask_range_plot], '.')
axes[2].axhline(y=table.iloc[i_plot]['range'], linestyle='--')
# axes[2].set_xlim([-8, -4])
axes[2].set_xlabel(r'$\beta\quad[Mm^{-1}]$')
# fig.suptitle(df.filename)
# axes[2].set_ylabel('Height (km)')
for ax in axes.flatten():
    ax.grid()
print(df.filename)
fig.savefig('F:/halo/paper/figures/' + df.filename + '_depo_profile.png', dpi=150,
            bbox_inches='tight')

# %%
table32 = table[(table['systemID'] == 32) & (pd.to_datetime(
    table[['year', 'month', 'day']]) >= '2017-11-22')]

co_cross_data = table32[['co_signal', 'cross_signal']].dropna()
H, co_edges, cross_edges = np.histogram2d(
    co_cross_data['co_signal'] - 1,
    co_cross_data['cross_signal'] - 1,
    bins=500)
X, Y = np.meshgrid(co_edges, cross_edges)
fig, ax = plt.subplots(figsize=(6, 4))
p = ax.pcolormesh(X, Y, H.T, norm=LogNorm())
ax.set_xlabel('co_SNR')
ax.set_ylabel('cross_SNR')
colorbar = fig5.colorbar(p, ax=ax)
colorbar.ax.set_ylabel('Number of observations')
ax.plot(co_cross_data['co_signal'] - 1,
        (co_cross_data['co_signal'] - 1) * 0.01,
        # label=r'$\frac{cross\_SNR}{co\_SNR} = 0.01$',
        linewidth=0.5)
ax.legend(loc='upper left')
fig.savefig('F:/halo/paper/figures/co_cross_saturation.png', dpi=150,
            bbox_inches='tight')

# %%
####################################
# Depo cloudbase
####################################
df = xr.open_dataset(r'F:\halo\classifier_new\46/2018-06-05-Hyytiala-46_classified.nc')
temp = df['depo_raw'].where(df['classified'] == 10)

# %%
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
                    np.log10(df['beta_raw']).T, cmap='jet', vmin=-8, vmax=-4)
p2 = ax3.pcolormesh(decimal_time, df['range'],
                    df['v_raw'].T, cmap='jet', vmin=-2, vmax=2)
p3 = ax5.pcolormesh(decimal_time, df['range'],
                    df['depo_bleed'].T, cmap='jet', vmin=0, vmax=0.5)
p4 = ax7.pcolormesh(decimal_time, df['range'],
                    df['classified'].T,
                    cmap=cmap, norm=norm)
for ax in [ax1, ax3, ax5, ax7]:
    # ax.yaxis.set_major_formatter(preprocess.m_km_ticks())
    ax.set_ylabel('Range [km, a.g.l]')

cbar = fig.colorbar(p1, ax=ax1)
cbar.ax.set_ylabel('Beta [' + units.get('beta_raw', None) + ']', rotation=90)
# cbar.ax.yaxis.set_label_position('left')
cbar = fig.colorbar(p2, ax=ax3)
cbar.ax.set_ylabel('Velocity [' + units.get('v_raw', None) + ']', rotation=90)
# cbar.ax.yaxis.set_label_position('left')
cbar = fig.colorbar(p3, ax=ax5)
cbar.ax.set_ylabel('Depolarization ratio')
# cbar.ax.yaxis.set_label_position('left')
cbar = fig.colorbar(p4, ax=ax7, ticks=[5, 15, 25, 35, 45])
cbar.ax.set_yticklabels(['Background', 'Aerosol',
                         'Precipitation', 'Clouds', 'Undefined'])
ax7.set_xlabel('Time [UTC - hour]')

fig.tight_layout()
