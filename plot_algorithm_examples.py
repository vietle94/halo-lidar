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
    ['white', '#2ca02c', 'red', 'gray'])
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


chosen_dates = ['2018-06-11', '2019-04-07', '2019-08-07', '2019-12-31']
files = glob.glob(r'F:\halo\paper\figures\algorithm\cloudnet/*.nc')
fig, axes = plt.subplots(4, 2, figsize=(12, 9), sharey=True)
for date_, file_cloudnet, ax in zip(chosen_dates, files, axes):

    df = xr.open_dataset(r'F:\halo\classifier_new\46/' + date_ + '-Hyytiala-46_classified.nc')
    df_cloudnet = xr.open_dataset(file_cloudnet)

    decimal_time = df['time'].dt.hour + \
        df['time'].dt.minute / 60 + df['time'].dt.second/3600
    p1 = ax[0].pcolormesh(decimal_time, df['range'],
                          df['classified'].T,
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
fig.savefig(path + 'algorithm_cloudnet.png', bbox_inches='tight')
