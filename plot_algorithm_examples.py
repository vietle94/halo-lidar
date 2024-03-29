import matplotlib.colors as colors
import copy
import datetime
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import glob
import numpy as np
from pathlib import Path
import matplotlib.dates as dates
import halo_data as hd
import xarray as xr
import string
from netCDF4 import Dataset
import glob

# %%

path = r'F:\halo\paper\figures\algorithm/'
####################################
# algorithm examples
####################################
cmap = mpl.colors.ListedColormap(
    # ['white', '#2ca02c', 'red', 'gray'])
    # ['white', '#009e73', '#e69f00', 'gray'])
    ['white', '#e69f00', '#56b4e9', 'gray'])
boundaries = [0, 10, 20, 40, 50]
norm = mpl.colors.BoundaryNorm(boundaries, cmap.N, clip=True)


_COLORS = {
    "green": "#3cb371",
    "darkgreen": "#253A24",
    "lightgreen": "#70EB5D",
    "yellowgreen": "#C7FA3A",
    "yellow": "#FFE744",
    "orange": "#ffa500",
    "pink": "#B43757",
    "red": "#F57150",
    "shockred": "#E64A23",
    "seaweed": "#646F5E",
    "seaweed_roll": "#748269",
    "white": "#ffffff",
    "lightblue": "#6CFFEC",
    "blue": "#209FF3",
    "skyblue": "#CDF5F6",
    "darksky": "#76A9AB",
    "darkpurple": "#464AB9",
    "lightpurple": "#6A5ACD",
    "purple": "#BF9AFF",
    "darkgray": "#2f4f4f",
    "lightgray": "#ECECEC",
    "gray": "#d3d3d3",
    "lightbrown": "#CEBC89",
    "lightsteel": "#a0b0bb",
    "steelblue": "#4682b4",
    "mask": "#C8C8C8",
}
cmap_cloudnet = mpl.colors.ListedColormap(
    [_COLORS["white"], _COLORS["lightblue"], _COLORS["blue"], _COLORS["purple"],
     _COLORS["lightsteel"], _COLORS["darkpurple"], _COLORS["orange"],
     _COLORS["yellowgreen"], _COLORS["lightbrown"], _COLORS["shockred"], _COLORS["pink"]])
boundaries_cloudnet = np.linspace(-0.5, 10.5, 12)
norm_cloudnet = mpl.colors.BoundaryNorm(boundaries_cloudnet, cmap_cloudnet.N, clip=True)


# chosen_dates = ['2018-06-11', '2019-04-07', '2019-08-07', '2019-12-31']
# files = glob.glob(r'F:\halo\paper\figures\algorithm\cloudnet/*.nc')
# fig, axes = plt.subplots(4, 2, figsize=(12, 9), sharey=True)
# for date_, file_cloudnet, ax in zip(chosen_dates, files, axes):
#
#     df = xr.open_dataset(r'F:\halo\classifier_new\46/' + date_ + '-Hyytiala-46_classified.nc')
#     df_cloudnet = xr.open_dataset(file_cloudnet)
#
#     decimal_time = df['time'].dt.hour + \
#         df['time'].dt.minute / 60 + df['time'].dt.second/3600
#     p1 = ax[0].pcolormesh(decimal_time, df['range'],
#                           df['classified'].T,
#                           cmap=cmap, norm=norm)
#
#     cbar = fig.colorbar(p1, ax=ax[0], ticks=[5, 15, 30, 45])
#     cbar.ax.set_yticklabels(['Background', 'Aerosol',
#                              'Hydrometeor', 'Undefined'])
#
#     decimal_time_cloudnet = pd.to_datetime(df_cloudnet['time']).hour + \
#         pd.to_datetime(df_cloudnet['time']).minute / 60 + \
#         pd.to_datetime(df_cloudnet['time']).second/3600
#     p2 = ax[1].pcolormesh(decimal_time_cloudnet,
#                           df_cloudnet['height'], df_cloudnet['target_classification'].T,
#                           cmap=cmap_cloudnet, norm=norm_cloudnet)
#     cbar = fig.colorbar(p2, ax=ax[1], ticks=np.arange(11))
#     cbar.ax.set_yticklabels(['Clear sky', 'Droplets', 'Drizzle or rain', 'Drizzle & droplets',
#                              'Ice', 'Ice & droplets', 'Melting ice', 'Melting & droplets',
#                              'Aerosols', 'Insects', 'Aerosol & Insects'])
#
# for n, ax in enumerate(axes.flatten()):
#     # ax.yaxis.set_major_formatter(preprocess.m_km_ticks())
#     ax.set_yticks([0, 4000, 8000])
#     ax.yaxis.set_major_formatter(hd.m_km_ticks())
#     ax.set_ylim(bottom=0)
#     ax.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
#             transform=ax.transAxes, size=12)
#     ax.set_xticks([0, 6, 12, 18, 24])
#     ax.set_xticklabels(['00:00', '06:00', '12:00', '18:00', '24:00'])
#     ax.set_xlabel('Time UTC')
#
# for ax in axes[:, 0]:
#     ax.set_ylabel('Height a.g.l [km]')
#
#
# fig.tight_layout()
# fig.savefig(path + 'algorithm_cloudnet.png', bbox_inches='tight')

# %%
# chosen_dates = ['2019-05-31']
# files = glob.glob(r'F:\halo\paper\figures\algorithm\cloudnet/' +
#                   chosen_dates[0].replace('-', '') + '*.nc')
# fig, ax = plt.subplots(1, 2, figsize=(12, 2.5), sharey=True)
# for date_, file_cloudnet in zip(chosen_dates, files):
#
#     df = xr.open_dataset(r'F:\halo\classifier_new\46/' + date_ + '-Hyytiala-46_classified.nc')
#     df_cloudnet = xr.open_dataset(file_cloudnet)
#
#     decimal_time = df['time'].dt.hour + \
#         df['time'].dt.minute / 60 + df['time'].dt.second/3600
#     p1 = ax[0].pcolormesh(decimal_time, df['range'],
#                           df['classified'].T,
#                           cmap=cmap, norm=norm)
#
#     cbar = fig.colorbar(p1, ax=ax[0], ticks=[5, 15, 30, 45])
#     cbar.ax.set_yticklabels(['Background', 'Aerosol',
#                              'Hydrometeor', 'Undefined'])
#
#     decimal_time_cloudnet = pd.to_datetime(df_cloudnet['time']).hour + \
#         pd.to_datetime(df_cloudnet['time']).minute / 60 + \
#         pd.to_datetime(df_cloudnet['time']).second/3600
#     p2 = ax[1].pcolormesh(decimal_time_cloudnet,
#                           df_cloudnet['height'], df_cloudnet['target_classification'].T,
#                           cmap=cmap_cloudnet, norm=norm_cloudnet)
#     cbar = fig.colorbar(p2, ax=ax[1], ticks=np.arange(11))
#     cbar.ax.set_yticklabels(['Clear sky', 'Droplets', 'Drizzle or rain', 'Drizzle & droplets',
#                              'Ice', 'Ice & droplets', 'Melting ice', 'Melting & droplets',
#                              'Aerosols', 'Insects', 'Aerosol & Insects'])
#
# for n, ax_ in enumerate(ax.flatten()):
#     # ax.yaxis.set_major_formatter(preprocess.m_km_ticks())
#     ax_.set_yticks([0, 4000, 8000])
#     ax_.yaxis.set_major_formatter(hd.m_km_ticks())
#     ax_.set_ylim(bottom=0)
#     ax_.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
#              transform=ax_.transAxes, size=12)
#     ax_.set_xticks([0, 6, 12, 18, 24])
#     ax_.set_xticklabels(['00:00', '06:00', '12:00', '18:00', '24:00'])
#     ax_.set_xlabel('Time UTC')
#
# ax[0].set_ylabel('Height a.g.l [km]')
# fig.tight_layout()
# fig.savefig('algorithm_cloudnet.png', bbox_inches='tight', dpi=500)

# %%
path = r'F:\halo\paper\figures\algorithm/'
chosen_dates = ['2018-04-08', '2018-06-11', '2018-09-03', '2019-12-31']
files = glob.glob(r'F:\halo\paper\figures\algorithm\cloudnet/*.nc')
fig, axes = plt.subplots(4, 2, figsize=(12, 9), sharey=True)
for date_, ax in zip(chosen_dates, axes):
    df = xr.open_dataset(r'F:\halo\classifier_new\46/' + date_ + '-Hyytiala-46_classified.nc')

    df_cloudnet = xr.open_dataset([x for x in files if date_.replace('-', '') in x][0])
    df_cloudnet['height'] = df_cloudnet['height'] - df_cloudnet['altitude'].values

    df_int = df.interp(range=df_cloudnet['height'], time=df_cloudnet['time'], method='nearest')
    decimal_time = df_int['time'].dt.hour + \
        df_int['time'].dt.minute / 60 + df_int['time'].dt.second/3600
    p1 = ax[0].pcolormesh(decimal_time, df_int['range'],
                          df_int['classified'].T,
                          cmap=cmap, norm=norm)

    cbar = fig.colorbar(p1, ax=ax[0], ticks=[5, 15, 30, 45])
    cbar.ax.set_yticklabels(['Background', 'Aerosol',
                             'Hydrometeor', 'Undefined'])

    decimal_time_cloudnet = pd.to_datetime(df_cloudnet['time']).hour + \
        pd.to_datetime(df_cloudnet['time']).minute / 60 + \
        pd.to_datetime(df_cloudnet['time']).second/3600
    p2 = ax[1].pcolormesh(decimal_time_cloudnet,
                          df_cloudnet['height'], df_cloudnet['target_classification'].T,
                          cmap=cmap_cloudnet, norm=norm_cloudnet)
    cbar = fig.colorbar(p2, ax=ax[1], ticks=np.arange(11))
    cbar.ax.set_yticklabels(['Clear sky', 'Droplets', 'Drizzle or rain', 'Drizzle & droplets',
                             'Ice', 'Ice & droplets', 'Melting ice', 'Melting & droplets',
                             'Aerosols', 'Insects', 'Aerosol & Insects'])
    cloudnet_aerosol = (df_cloudnet['target_classification'] == 8) | \
        (df_cloudnet['target_classification'] == 10)
    cloudnet_hydro = (df_cloudnet['target_classification'] < 8) * \
        (df_cloudnet['target_classification'] > 0)

    ai_aerosol = df_int['classified'] == 10
    same_aerosol = np.sum((cloudnet_aerosol * ai_aerosol).values)
    false_aerosol = np.sum((cloudnet_hydro * ai_aerosol).values)

    total_cloudnet_aerosol = np.sum(cloudnet_aerosol.values)
    total_ai_aerosol = np.sum(ai_aerosol.values)

    print(same_aerosol, total_cloudnet_aerosol, total_ai_aerosol,
          false_aerosol/total_ai_aerosol,
          (total_cloudnet_aerosol-same_aerosol)/total_ai_aerosol)
for n, ax in enumerate(axes.flatten()):
    # ax.yaxis.set_major_formatter(preprocess.m_km_ticks())
    ax.set_yticks([0, 4000, 8000])
    ax.yaxis.set_major_formatter(hd.m_km_ticks())
    ax.set_ylim(bottom=0)
    ax.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
            transform=ax.transAxes, size=12)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xticklabels(['00:00', '06:00', '12:00', '18:00', '24:00'])
    ax.set_xlabel('Time UTC')

for ax in axes[:, 0]:
    ax.set_ylabel('Height a.g.l [km]')

fig.tight_layout()
fig.savefig(path + 'algorithm_cloudnet.png', bbox_inches='tight', dpi=500)

###############################
# %% Collect overall statistics
###############################
# cloudnet_files = glob.glob(r'F:\halo\paper\figures\algorithm\cloudnet/all/*.nc')
# ai_files = glob.glob(r'F:\halo\classifier_new\46/*.nc')
# chosen_dates = [x.split('-Hyy')[0].split('\\')[-1] for x in ai_files]
#
# for date_ in chosen_dates:
#     df = xr.open_dataset(r'F:\halo\classifier_new\46/' + date_ + '-Hyytiala-46_classified.nc')
#     try:
#         df_cloudnet = xr.open_dataset([x for x in cloudnet_files if date_.replace('-', '') in x][0])
#     except IndexError:
#         continue
#     df_cloudnet['height'] = df_cloudnet['height'] - df_cloudnet['altitude'].values
#     df_int = df.interp(range=df_cloudnet['height'], time=df_cloudnet['time'], method='nearest')
#
#     cloudnet_aerosol = (df_cloudnet['target_classification'] == 8) | \
#         (df_cloudnet['target_classification'] == 10)
#     cloudnet_hydro = (df_cloudnet['target_classification'] < 8) * \
#         (df_cloudnet['target_classification'] > 0)
#
#     ai_aerosol = df_int['classified'] == 10
#     same_aerosol = np.sum((cloudnet_aerosol * ai_aerosol).values)
#     false_aerosol = np.sum((cloudnet_hydro * ai_aerosol).values)
#
#     total_cloudnet_aerosol = np.sum(cloudnet_aerosol.values)
#     total_ai_aerosol = np.sum(ai_aerosol.values)
#     result = pd.DataFrame.from_records([{
#         'time': date_,
#         'ai_aerosol': total_ai_aerosol,
#         'cloudnet_aerosol': total_cloudnet_aerosol,
#         'same_aerosol': same_aerosol,
#         'false_aerosol': false_aerosol,
#         'false_aerosol_percent': false_aerosol/total_ai_aerosol,
#         'miss_aerosol': (total_cloudnet_aerosol-same_aerosol),
#         'miss_aerosol_percent': (total_cloudnet_aerosol-same_aerosol)/total_ai_aerosol
#     }])
#     with open(r'F:\halo\paper\figures\algorithm\cloudnet/ai_summary.csv', 'a') as f:
#         result.to_csv(f, header=f.tell() == 0, index=False)


#############################
# %% Overall statistics
#############################
df = pd.read_csv(r'F:\halo\paper\figures\algorithm\cloudnet/ai_summary.csv')

# %%
false_per = np.sum(df['false_aerosol'])/np.sum(df['ai_aerosol'])
miss_per = np.sum(df['miss_aerosol'])/np.sum(df['ai_aerosol'])

false_per
miss_per
