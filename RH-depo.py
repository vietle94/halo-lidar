import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import halo_data as hd
from scipy.stats import binned_statistic

%matplotlib qt

# %%
df = hd.halo_data('F:/halo/46/depolarization/20180801_fmi_halo-doppler-lidar-46-depolarization.nc')
df.unmask999()
df.filter_height()
df.filter(variables=['beta_raw', 'depo_raw'],
          ref='co_signal',
          threshold=1 + 3 * df.snr_sd)

# %%
weather = pd.read_csv('F:/weather/hyytiala_2018_2.csv')
weather = weather.rename(columns={'Vuosi': 'year', 'Kk': 'month',
                                  'Pv': 'day', 'Klo': 'time'})
weather[['year', 'month', 'day']] = weather[['year',
                                             'month', 'day']].astype(str)
weather['month'] = weather['month'].str.zfill(2)
weather['day'] = weather['day'].str.zfill(2)
weather['datetime'] = weather['year'] + weather['month'] + \
    weather['day'] + weather['time']
weather['datetime'] = pd.to_datetime(weather['datetime'], format='%Y%m%d%H:%M')
weather = weather[weather['datetime'].dt.date == pd.to_datetime(df.date)]
weather['hour'] = weather['datetime'].dt.hour + \
    weather['datetime'].dt.minute / 60
# %%
co = df.data['co_signal'][:, df.data['range'] < 300].mean(axis=1)
cross = df.data['cross_signal'][:, df.data['range'] < 300].mean(axis=1)
co_bin = binned_statistic(df.data['time'], co, bins=np.arange(25),
                          statistic=np.nanmean)
cross_bin = binned_statistic(df.data['time'], cross, bins=np.arange(25),
                             statistic=np.nanmean)
dep = (cross_bin.statistic - 1) / (co_bin.statistic - 1)
t = np.arange(24) + 0.5
# %%
plt.rcParams.update({'font.size': 12})
fig = plt.figure(figsize=(11, 10))
gs = fig.add_gridspec(4, 2, width_ratios=[60, 1])
ax0 = fig.add_subplot(gs[0, 0])
cax0 = fig.add_subplot(gs[0, 1])
ax1 = fig.add_subplot(gs[1, 0])
cax1 = fig.add_subplot(gs[1, 1])
ax2 = fig.add_subplot(gs[2, 0])
ax3 = fig.add_subplot(gs[3, 0])
p0 = ax0.pcolormesh(df.data['time'], df.data['range'],
                    np.log10(df.data['beta_raw']).T, vmin=-8, vmax=-4, cmap='jet')
p1 = ax1.pcolormesh(df.data['time'], df.data['range'],
                    df.data['depo_raw'].T, vmin=0, vmax=0.5, cmap='jet')
cbar = fig.colorbar(p0, cax=cax0)
cbar.ax.set_ylabel('Attenuated backscatter')
cbar.ax.yaxis.set_label_position('left')
cbar = fig.colorbar(p1, cax=cax1)
cbar.ax.set_ylabel('Depolarization ratio')
cbar.ax.yaxis.set_label_position('left')
for ax_ in [ax1, ax0]:
    ax_.yaxis.set_major_formatter(hd.m_km_ticks())
    ax_.set_ylabel('Height [km, a.g.l]')
ax2.plot(weather['hour'], weather['Suhteellinen kosteus (%)'], '.')
ax2.set_ylabel('Relative humidity [%]')
ax3.plot(t, dep, '.')
ax3.set_xlim([0, 24])
ax3.set_xlabel('Time UTC [hour]')
ax3.set_ylabel('Hourly depo < 300m')
fig.tight_layout()
fig.savefig('F:/halo/classifier/RH/' + df.filename + '_rh.png',
            bbox_inches='tight', dpi=150)

# %%
weather_ = weather[weather['datetime'].dt.hour > 8]
RH = weather_.groupby(weather_['datetime'].dt.hour)['Suhteellinen kosteus (%)'].mean()

# %%
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(RH, dep[t > 8.5], '.')
ax.set_xlabel('Relative humidity [%]')
ax.set_ylabel('Hourly depo < 300m')
fig.savefig('F:/halo/classifier/RH/' + df.filename + '_rh_scatter.png',
            bbox_inches='tight', dpi=150)

# %%
