import json
from sklearn.metrics import r2_score
import dask.array as da
import matplotlib.cm as cm
import calendar
from scipy.stats import binned_statistic_2d
import os
from scipy.ndimage import uniform_filter
from sklearn.cluster import DBSCAN
import matplotlib.colors as colors
from scipy.ndimage import maximum_filter
from scipy.ndimage import median_filter
import matplotlib
import copy
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import datetime
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from sklearn.mixture import GaussianMixture
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
import scipy.stats as stats
from netCDF4 import Dataset
import pywt
%matplotlib qt

# %%


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


def wavelet_denoising(x, wavelet='db4', level=1):
    coeff = pywt.wavedec(x, wavelet, mode="symmetric")
    # sigma = (1/0.6745) * madev(coeff[-level])
    # uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    uthresh = np.median(np.abs(coeff[1]))/0.6745 * np.sqrt(2 * np.log(len(coeff[1])))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode='symmetric')


# %%
df = xr.open_dataset(r'F:\halo\classifier_new\46/2018-04-15-Hyytiala-46_classified.nc')
df = bleed_through(df)

# %%
total_depo = []
total_depo_corrected = []
filter_aerosol = df.classified == 10
avg = df[['co_signal', 'cross_signal_bleed']].resample(time='60min').mean(dim='time')
avg['aerosol_percentage'] = filter_aerosol.resample(time='60min').mean(dim='time')
for time in range(24):
    co = avg['co_signal'][time, :].values
    cross = avg['cross_signal_bleed'][time, :].values

    filtered = wavelet_denoising(co, wavelet='bior4.4', level=1)
    filtered = filtered[:len(co)]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(co, label='Raw')
    ax.plot(filtered, label='Filtered')
    ax.axhline(y=6e-5)
    ax.set_ylim([-0.0002+1, 0.0003+1])
    ax.legend()
    ax.set_title(f"DWT Denoising with {'bior4.4'} Wavelet", size=15)
    fig.savefig('F:/halo/paper/figures/background_correction/denoise_' + str(time),
                bbox_inches='tight')

    background = filtered < 1+6e-5

    selected_range = df['range'][background]
    selected_co = co[background]
    selected_cross = cross[background]

    a, b, c = np.polyfit(selected_range, selected_co, deg=2)
    y_co = c + b*df['range'] + a*(df['range']**2)
    y_co_background = c + b*selected_range + a*(selected_range**2)

    a, b, c = np.polyfit(selected_range, selected_cross, deg=2)
    y_cross = c + b*df['range'] + a*(df['range']**2)
    y_cross_background = c + b*selected_range + a*(selected_range**2)

    fig, axes = plt.subplots(1, 3, sharex=True, sharey=True,
                             figsize=(9, 4))

    axes[0].plot(co, df['range'], '.', label='$SNR_{co}$')
    axes[0].plot(cross, df['range'], '.', label='$SNR_{cross}$')

    axes[2].plot(selected_co, selected_range, '+', label='$SNR_{co}$')
    axes[2].plot(selected_cross, selected_range, '+', label='$SNR_{cross}$')

    axes[0].plot(y_co, df['range'], label='Fitted $SNR_{co}$')
    axes[0].plot(y_cross, df['range'], label='Fitted $SNR_{cross}$')

    axes[1].plot((co)/y_co, df['range'], '.', label='Corrected $SNR_{co}$')
    axes[1].plot((cross)/y_cross, df['range'], '.', label='Corrected $SNR_{cross}$')

    axes[0].set_xlim([0.9995, 1.001])
    axes[0].yaxis.set_major_formatter(hd.m_km_ticks())
    axes[0].set_ylabel('Height a.g.l [km]')
    for ax in axes.flatten():
        ax.tick_params(axis='x', labelrotation=45)
        ax.legend()
        ax.set_xlabel('SNR')
    fig.savefig('F:/halo/paper/figures/background_correction/correction_' + str(time),
                bbox_inches='tight')

    mask = (avg.aerosol_percentage[time, :] > 0.8) & (avg.range < 2000)
    co[mask]
    depo = (cross - 1) / (co - 1)
    depo = depo[mask]
    depo_corrected = (cross/y_cross - 1) / (co/y_co - 1)
    depo_corrected = depo_corrected[mask]
    total_depo.append(depo)
    total_depo_corrected.append(depo_corrected.values)

# %%
np.array(total_depo).flatten().flatten()
fig, ax = plt.subplots(figsize=(6, 4), sharex=True, sharey=True)
ax.hist(depo_corrected[df.range.values < 1500].values, label='Corrected depo')
ax.hist(depo[df.range.values < 1500], alpha=0.5, label='Not corrected depo')
ax.legend()
ax.set_xlabel('$\delta$')
ax.set_ylabel('N')
fig.savefig('temp.png', bbox_inches='tight')

# %%

plt.hist(depo_corrected[df.range < 1500], bins=20)
# %%


def gaussian(x, s):
    return 1./np.sqrt(2. * np.pi * s**2) * np.exp(-x**2 / (2. * s**2))


# %%
gaus = np.array([gaussian(x, 1) for x in range(-13, 14, 1)])

# %%
coef = pywt.wavedec(np.convolve(co, gaus, 'same'), 'haar', level=1)
fig, ax = plt.subplots(len(coef) + 1, 1, figsize=(9, 6))
ax[0].plot(co)
ax[1].plot(pywt.idwt(coef[0], None, 'haar'))
for i in range(1, len(coef)):
    ax[i+1].plot(pywt.idwt(None, coef[i], 'haar'))


# %%


def lowpassfilter(signal, thresh=0.63, wavelet="haar"):
    thresh = thresh*np.nanmax(signal)
    coeff = pywt.wavedec(signal, wavelet, mode="per")
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft") for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per")
    return reconstructed_signal

# def lowpassfilter(signal, thresh = 0.63, wavelet="haar"):
#     thresh = thresh*np.nanstd(signal)
#     coeff = pywt.wavedec(signal, wavelet, mode="per" )
#     coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft" ) for i in coeff[1:])
#     reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per" )
#     return reconstructed_signal


fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(co-1, color="b", alpha=0.5, label='original signal')
rec = lowpassfilter(co-1, 0.1)
ax.plot(rec, 'k', label='DWT smoothing}', linewidth=2)
ax.legend()
ax.set_title('Removing High Frequency Noise with DWT', fontsize=18)
ax.set_ylabel('Signal Amplitude', fontsize=16)
ax.set_xlabel('Sample No', fontsize=16)
plt.show()

# %%
fig, ax = plt.subplots()
ax.plot(df['range'], co-1)
ax.plot(df['range'], np.convolve(co-1, [0, 1, -1, 0], 'same'))

# %%


def gaussian(x, s):
    return 1./np.sqrt(2. * np.pi * s**2) * np.exp(-x**2 / (2. * s**2))


# %%
gaus = np.array([gaussian(x, 1) for x in range(-3, 4, 1)])
fig, ax = plt.subplots()
ax.plot(df['range'], co-1)
ax.plot(df['range'], np.convolve(co-1, gaus, 'same'))
ax.plot(df['range'], np.convolve(np.convolve(co-1, gaus, 'same'), [0, 1, -1, 0], 'same'))


# %%
plt.plot(np.convolve(co-1, [0, 1, -1, 0]))
plt.plot(co-1)

# %%
coef = pywt.wavedec(co, 'sym6', level=1)
threshold = np.median(np.abs(coef[1]))/0.6745 * np.sqrt(2 * len(coef[1]))
coef[1:] = (pywt.threshold(i, value=threshold, mode="soft") for i in coef[1:])
reconstructed_signal = pywt.waverec(coef, 'sym6')

# %%
plt.plot(reconstructed_signal)
plt.plot(co)
reconstructed_signal.size
co.size

# %%


def madev(d, axis=None):
    """ Mean absolute deviation of a signal """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def wavelet_denoising(x, wavelet='db4', level=1):
    coeff = pywt.wavedec(x, wavelet, mode="symmetric")
    sigma = (1/0.6745) * madev(coeff[-level])
    # uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    uthresh = np.median(np.abs(coeff[1]))/0.6745 * np.sqrt(2 * np.log(len(coeff[1])))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode='symmetric')


# %%
# for wav in pywt.wavelist():
for wav in ['sym5', 'db7', 'db5', 'bior4.4']:
    try:
        filtered = wavelet_denoising(co, wavelet=wav, level=1)
    except:
        pass

    plt.figure(figsize=(10, 6))
    plt.plot(co, label='Raw')
    plt.plot(filtered, label='Filtered')
    plt.axhline(y=0)

    plt.ylim([-0.0002, 0.0003])
    plt.legend()
    plt.title(f"DWT Denoising with {wav} Wavelet", size=15)
    plt.show()

# %%
