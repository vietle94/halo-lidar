from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as mtick
from sklearn.metrics import r2_score
import string
import datetime
import calendar
from matplotlib.colors import LogNorm
import xarray as xr
from pathlib import Path
import glob
import halo_data as hd
from scipy.stats import binned_statistic_2d
import statsmodels.api as sm
from decimal import Decimal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

%matplotlib qt

# %%
df_ref = pd.read_csv(r'C:\Users\le\OneDrive - University of Helsinki/depo_wave.csv')
path = r'F:\halo\paper\figures\depo_aerosol/'

# %%
my_cmap = plt.cm.get_cmap('viridis')
wave_color = {'355nm': my_cmap(0.9), '532nm': my_cmap(0.75), '710nm': my_cmap(0.6),
              '1064nm': my_cmap(0.4), '1565nm': my_cmap(0.2)}
# format_type =
# %%
df_ref['Type'] = ['dust' if 'dust' in x else x for x in df_ref['Type']]
df_ref['Type'] = ['marine' if 'marine' in x else x for x in df_ref['Type']]
df_ref['Type'] = ['smoke' if 'smoke' in x else x for x in df_ref['Type']]
df_ref['Type'] = ['anthropogenic\npollution' if 'urban' in x else x for x in df_ref['Type']]
df_ref['Type'] = ['pollen' if 'pollen' in x else x for x in df_ref['Type']]
df_ref['Type'] = ['volcanic\nash' if 'volcanic ash' in x else x for x in df_ref['Type']]
df_ref['Wavelength'] = [int(x[:-2]) if 'nm' in x else x for x in df_ref['Wavelength']]
df_ref['color'] = [wave_color[str(i) + 'nm'] for i in df_ref['Wavelength']]

# %%
df_ref = df_ref.sort_values(['Type', 'Wavelength']).reset_index(drop='True')

# %%
i_start = 0
facecolor = 0
aerosol_type = 'dust'
fig, ax = plt.subplots(figsize=(9, 4))
for i, val in df_ref.iterrows():
    if val['Depol_type'] == 'Range':
        ax.vlines(i, ymin=val['Depol0'], ymax=val['Depol1'],
                  color=val['color'])
    else:
        ax.errorbar(i, val['Depol0'], val['Depol1'], fmt='.',
                    elinewidth=1, c=val['color'])
    if aerosol_type != val['Type']:
        ax.text((i_start + i)/2 - 2, 0.6, aerosol_type.upper(), va='center',
                bbox=dict(facecolor='white', edgecolor='black', alpha=1), size=8)
        if facecolor % 2 == 1:
            ax.axvspan(i_start, i - 0.5, alpha=0.1, facecolor='0.2')
        facecolor += 1
        aerosol_type = val['Type']
        i_start = i - 0.5
if facecolor % 2 == 1:
    ax.text((i_start + i)/2 - 2, 0.6, aerosol_type.upper(), va='center',
            bbox=dict(facecolor='white', edgecolor='black', alpha=1), size=8)
    ax.axvspan(i_start, i+0.5, alpha=0.1, facecolor='0.2')
ax.grid()
legend_elements = [Line2D([], [], label='355nm', color=wave_color['355nm'], linewidth=1.5, marker='o',
                          markerfacecolor=wave_color['355nm'], markeredgewidth=0, markersize=7),
                   Line2D([], [], label='532nm', color=wave_color['532nm'], linewidth=1.5, marker='o',
                          markerfacecolor=wave_color['532nm'], markeredgewidth=0, markersize=7),
                   Line2D([], [], label='710nm', color=wave_color['710nm'], linewidth=1.5, marker='o',
                          markerfacecolor=wave_color['710nm'], markeredgewidth=0, markersize=7),
                   Line2D([], [], label='1064nm', color=wave_color['1064nm'], linewidth=1.5, marker='o',
                          markerfacecolor=wave_color['1064nm'], markeredgewidth=0, markersize=7),
                   Line2D([], [], label='1565nm', color=wave_color['1565nm'], linewidth=1.5, marker='o',
                          markerfacecolor=wave_color['1565nm'], markeredgewidth=0, markersize=7)]
fig.legend(handles=legend_elements, ncol=5, loc='upper center')

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

weather_full = pd.DataFrame()
for grp_lab, grp_val in weather.groupby('location2'):
    weather_ = grp_val.set_index('datetime').resample('1H').mean().reset_index()
    weather_['location'] = grp_lab
    weather_full = weather_full.append(weather_, ignore_index=True)

df_full['datetime'] = df_full['time']
df = pd.merge(weather_full, df_full)

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
jitter = {}
for place, v in zip(['Uto', 'Hyytiala', 'Vehmasmaki', 'Sodankyla'],
                    np.linspace(-0.15, 0.15, 4)):
    jitter[place] = v
# fig, ax = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
fig = plt.figure(figsize=(12, 4))
gs = fig.add_gridspec(nrows=1, ncols=6)
ax0 = fig.add_subplot(gs[0, :2])
ax1 = fig.add_subplot(gs[0, 2:6], sharey=ax0)
ax = [ax0, ax1]
group = df_miss[(df_miss['count'] > 15) & (
    df_miss['depo_corrected_sd'] < 0.05)].groupby(['location'])
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
                   label=k, marker='.',
                   fmt='--', elinewidth=1, linewidth=0.5)

    ax[0].set_xticks(np.arange(1, 13, 3))
    ax[0].set_xticklabels(['Jan', 'April', 'July', 'Oct'])
    ax[0].set_ylabel('$\delta$')
    ax[0].grid(axis='x', which='major', linewidth=0.5, c='silver')
    ax[0].grid(axis='y', which='major', linewidth=0.5, c='silver')
    # ax[0].set_xlabel('Month')
    ax[0].legend()


i_start = -8
facecolor = 0
aerosol_type = 'anthropogenic\npollution'
for i, val in df_ref.iterrows():
    if val['Depol_type'] == 'Range':
        ax[1].vlines(i, ymin=val['Depol0'], ymax=val['Depol1'],
                     color=val['color'])
    else:
        ax[1].errorbar(i, val['Depol0'], val['Depol1'], fmt='.',
                       elinewidth=1, c=val['color'])
    if aerosol_type != val['Type']:
        ax[1].text((i_start + i)/2 - 3, 0.47, aerosol_type.capitalize(), va='center',
                   bbox=dict(facecolor='white', edgecolor='black', alpha=1), size=8)
        if facecolor % 2 == 1:
            ax[1].axvspan(i_start, i - 0.5, alpha=0.1, facecolor='0.2')
        facecolor += 1
        aerosol_type = val['Type']
        i_start = i - 0.5
if facecolor % 2 == 1:
    ax[1].text((i_start + i)/2 - 3, 0.47, aerosol_type.capitalize(), va='center',
               bbox=dict(facecolor='white', edgecolor='black', alpha=1), size=8)
    ax[1].axvspan(i_start, i+0.5, alpha=0.1, facecolor='0.2')
ax[1].grid()
legend_elements = [Line2D([], [], label='355nm', color=wave_color['355nm'], linewidth=1.5, marker='.',
                          markerfacecolor=wave_color['355nm'], markeredgewidth=0, markersize=7),
                   Line2D([], [], label='532nm', color=wave_color['532nm'], linewidth=1.5, marker='.',
                          markerfacecolor=wave_color['532nm'], markeredgewidth=0, markersize=7),
                   Line2D([], [], label='710nm', color=wave_color['710nm'], linewidth=1.5, marker='.',
                          markerfacecolor=wave_color['710nm'], markeredgewidth=0, markersize=7),
                   Line2D([], [], label='1064nm', color=wave_color['1064nm'], linewidth=1.5, marker='.',
                          markerfacecolor=wave_color['1064nm'], markeredgewidth=0, markersize=7),
                   Line2D([], [], label='1565nm', color=wave_color['1565nm'], linewidth=1.5, marker='.',
                          markerfacecolor=wave_color['1565nm'], markeredgewidth=0, markersize=7)]
ax[1].legend(handles=legend_elements, ncol=5, loc='upper center',
             bbox_to_anchor=(0.52, 1.15))
plt.setp(ax[1].get_yticklabels(), visible=False)
ax[1].set_xlabel('Reference number (see table xx)')
# fig.subplots_adjust(wspace=1)
for n, ax_ in enumerate(ax):
    ax_.text(0, 1.05, '(' + string.ascii_lowercase[n] + ')',
             transform=ax_.transAxes, size=12)
ax[0].set_ylim(top=0.5)
# fig.savefig(path + '/depo_vs_ref.png', bbox_inches='tight', dpi=1000)

# %%
df_ref[df_ref['Type'].isin(['anthropogenic\npollution', 'smoke'])]

df_ref[df_ref['Type'] == 'pollen']
