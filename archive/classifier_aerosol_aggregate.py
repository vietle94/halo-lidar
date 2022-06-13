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
import json
import dask.array as da
import glob
import matplotlib.dates as mdates
%matplotlib qt

# %%
#############
# 1hour averaged
#############


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
site_ = ['32', '46, ' '33', '53', '54']
for site in site_:
    for file in glob.glob('F:/halo/classifier_new/' + site + '/*.nc'):
        df = xr.open_dataset(file)
        df = bleed_through(df)
        df = df.loc[{'time': sorted(df.coords['time'].values)}]

        filter_co = df.co_signal > 1 + 3 * df.attrs['background_snr_sd']
        df['cross_signal_bleed'] = df.cross_signal_bleed.where(filter_co)
        df['classified'] = df.classified.where(filter_co)
        df['co_signal'] = df.co_signal.where(filter_co)

        filter_aerosol = df.classified == 10

        hour1 = df[['co_signal', 'cross_signal_bleed']].where(
            filter_aerosol).resample(time='60min').mean()
        hour1['aerosol_percentage'] = filter_aerosol.resample(time='60min').mean()
        if len(hour1.time) == 0:
            continue
        hour1.to_netcdf('F:/halo/depo_aerosol/' + site + '/' + df.attrs['file_name'] +
                        '_classified.nc', format='NETCDF3_CLASSIC')
