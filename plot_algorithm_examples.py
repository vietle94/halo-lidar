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

# %%

path = r'F:\halo\paper\figures\algorithm/'
####################################
# algorithm examples
####################################
chosen_dates = ['2018-06-11', '2019-04-07', '2019-08-07', '2019-12-31']
for date_ in chosen_dates:

    df = xr.open_dataset(r'F:\halo\classifier_new\46/' + date_ + '-Hyytiala-46_classified.nc')
    data = hd.getdata('F:/halo/46/depolarization/')

    date = date_.replace('-', '')
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
        ['white', '#2ca02c', 'red', 'gray'])
    boundaries = [0, 10, 20, 40, 50]
    norm = mpl.colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    decimal_time = df['time'].dt.hour + \
        df['time'].dt.minute / 60 + df['time'].dt.second/3600
    fig, ax = plt.subplots(4, 1, figsize=(10, 8), sharex=True, sharey=True)
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
    for n, ax in enumerate([ax1, ax3, ax5, ax7]):
        # ax.yaxis.set_major_formatter(preprocess.m_km_ticks())
        ax.set_ylabel('Height a.g.l [km]')
        ax.set_yticks([0, 4000, 8000])
        ax.yaxis.set_major_formatter(hd.m_km_ticks())
        ax.set_ylim(bottom=0)
        ax.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
                transform=ax.transAxes, size=12)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_xticklabels(['00:00', '06:00', '12:00', '18:00', '24:00'])
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
    cbar = fig.colorbar(p4, ax=ax7, ticks=[5, 15, 30, 45])
    cbar.ax.set_yticklabels(['Background', 'Aerosol',
                             'Hydrometeor', 'Undefined'])
    ax7.set_xlabel('Time UTC')

    fig.tight_layout()
    fig.savefig(path + 'algorithm_' + df_.filename + '.png', bbox_inches='tight')
