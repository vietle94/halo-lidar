from scipy.optimize import least_squares
from scipy.optimize import leastsq
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
from scipy.signal import find_peaks, peak_widths
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


# %%
df = xr.open_dataset(r'F:\halo\classifier_new\46/2018-04-15-Hyytiala-46_classified.nc')
df = bleed_through(df)

# %%
time = 0
co = avg['co_signal'][time, :].values
cross = avg['cross_signal_bleed'][time, :].values

filtered = wavelet_denoising(co-1, wavelet='bior1.1', level=2)
filtered = filtered[:len(co)]
filtered = filtered + 1
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(co, label='Raw')
ax.plot(filtered, label='Filtered')
ax.axhline(y=1 + 6e-5)
ax.set_ylim([-0.0004+1, 0.0006+1])
ax.legend()
ax.set_title(f"DWT Denoising with {'bior4.4'} Wavelet", size=15)

# %%
time = 0
co = avg['co_signal'][time, :].values
cross = avg['cross_signal_bleed'][time, :].values
# for wav in pywt.wavelist():
for wav in ['db14', 'db13', 'db5', 'db15', 'db17', 'rbio1.5', 'rbio1.1', 'morl', 'mexh', 'haar', 'bior2.6', 'bior1.1']:
    wavelet = wav
    try:
        coeff = pywt.swt(np.pad(co-1, (0, 67), 'constant',
                                constant_values=(0, 0)), wavelet, level=5)
        uthresh = np.median(np.abs(coeff[1]))/0.6745 * np.sqrt(2 * np.log(len(coeff[1])))
        coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
        filtered = pywt.iswt(coeff, wavelet) + 1
    except:
        pass

    fig, ax = plt.subplots()
    ax.plot(co)
    ax.plot(filtered[:len(co)])
    ax.set_title(f"DWT Denoising with {wav} Wavelet", size=15)
    ax.set_ylim([-0.0004+1, 0.0006+1])
    ax.axhline(y=1 + 6e-5)


# %%
total_depo = np.array([])
total_depo_corrected = np.array([])
filter_aerosol = df.classified == 10
wavelet = 'bior2.6'
avg = df[['co_signal', 'cross_signal_bleed']].resample(time='60min').mean(dim='time')
avg['aerosol_percentage'] = filter_aerosol.resample(time='60min').mean(dim='time')
for time in range(24):
    co = avg['co_signal'][time, :].values
    cross = avg['cross_signal_bleed'][time, :].values

    coeff = pywt.swt(np.pad(co-1, (0, 67), 'constant', constant_values=(0, 0)), wavelet, level=5)
    uthresh = np.median(np.abs(coeff[1]))/0.6745 * np.sqrt(2 * np.log(len(coeff[1])))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    filtered = pywt.iswt(coeff, wavelet) + 1

    filtered = filtered[:len(co)]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(co, label='Raw')
    ax.plot(filtered, label='Filtered')
    ax.axhline(y=1+6e-5)
    ax.set_ylim([-0.0004+1, 0.0006+1])

    ax.legend()
    ax.set_title(f"SWT Denoising with {wavelet} Wavelet", size=15)
    # fig.savefig('F:/halo/paper/figures/background_correction2/denoise_' + str(time),
    #             bbox_inches='tight')

    background = filtered < 1+6e-5

    selected_range = df['range'][background]
    selected_co = co[background]
    selected_cross = cross[background]

    a, b, c = myfit(selected_range, selected_co)
    y_co = c + b*df['range'] + a*(df['range']**2)
    y_co_background = c + b*selected_range + a*(selected_range**2)

    a, b, c = myfit(selected_range, selected_cross)
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
    # fig.savefig('F:/halo/paper/figures/background_correction2/correction_' + str(time),
    #             bbox_inches='tight')

    mask = (avg.aerosol_percentage[time, :] > 0.8) & (avg.range < 2000)
    co[mask]
    depo = (cross - 1) / (co - 1)
    depo = depo[mask]
    depo_corrected = (cross/y_cross - 1) / (co/y_co - 1)
    depo_corrected = depo_corrected[mask]
    total_depo = np.append(depo, total_depo)
    total_depo_corrected = np.append(depo_corrected.values, total_depo_corrected)

# %%
fig, ax = plt.subplots(figsize=(6, 4), sharex=True, sharey=True)
ax.hist(total_depo[(total_depo < 0.5) & (total_depo > -0.1)], label='Not corrected depo', bins=20)
ax.hist(total_depo_corrected[(total_depo_corrected < 0.5) & (total_depo_corrected > -0.1)],
        alpha=0.5, label='Corrected depo', bins=20)
ax.legend()
ax.set_xlim([-0.01, 0.2])
ax.set_xlabel('$\delta$')
ax.set_ylabel('N')
fig.savefig('F:/halo/paper/figures/background_correction2/summary', bbox_inches='tight')

# %%
total_depo = np.array([])
total_depo_corrected = np.array([])
filter_aerosol = df.classified == 10
wavelet = 'bior2.6'
avg = df[['co_signal', 'cross_signal_bleed']].resample(time='60min').mean(dim='time')
avg['aerosol_percentage'] = filter_aerosol.resample(time='60min').mean(dim='time')
for time in range(24):
    co = avg['co_signal'][time, :].values
    cross = avg['cross_signal_bleed'][time, :].values

    coeff = pywt.swt(np.pad(co-1, (0, 67), 'constant', constant_values=(0, 0)), wavelet, level=5)
    uthresh = np.median(np.abs(coeff[1]))/0.6745 * np.sqrt(2 * np.log(len(coeff[1])))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    filtered = pywt.iswt(coeff, wavelet) + 1
    filtered = filtered[:len(co)]

    coeff = pywt.swt(np.pad(cross-1, (0, 67), 'constant', constant_values=(0, 0)), wavelet, level=5)
    uthresh = np.median(np.abs(coeff[1]))/0.6745 * np.sqrt(2 * np.log(len(coeff[1])))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    filtered_cross = pywt.iswt(coeff, wavelet) + 1
    filtered_cross = filtered_cross[:len(cross)]

    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    ax[0].plot(co, label='Raw co')
    ax[0].plot(filtered, label='Filtered co')
    ax[0].legend()
    ax[1].plot(cross, label='Raw cross')
    ax[1].plot(filtered_cross, label='Filtered cross')
    ax[1].legend()
    fig.suptitle(f"SWT Denoising with {wavelet} Wavelet", size=15)
    fig.savefig('F:/halo/paper/figures/background_correction2/denoise_co_cross' + str(time),
                bbox_inches='tight')

    background = filtered < 1+6e-5

    selected_range = df['range'][background]
    selected_co = filtered[background]
    selected_cross = filtered_cross[background]

    a, b, c = myfit(selected_range, selected_co)
    y_co = c + b*df['range'] + a*(df['range']**2)
    y_co_background = c + b*selected_range + a*(selected_range**2)

    a, b, c = myfit(selected_range, selected_cross)
    y_cross = c + b*df['range'] + a*(df['range']**2)
    y_cross_background = c + b*selected_range + a*(selected_range**2)

    fig, axes = plt.subplots(1, 3, sharex=True, sharey=True,
                             figsize=(9, 4))

    axes[0].plot(co, df['range'], '.', label='$SNR_{co}$')
    axes[0].plot(cross, df['range'], '.', label='$SNR_{cross}$')

    axes[2].plot(filtered, df['range'], '.', label='$SNR_{co}$')
    axes[2].plot(filtered_cross, df['range'], '.', label='$SNR_{cross}$')

    axes[0].plot(y_co, df['range'], label='Fitted $SNR_{co}$')
    axes[0].plot(y_cross, df['range'], label='Fitted $SNR_{cross}$')

    axes[1].plot((filtered)/y_co, df['range'], '.', label='Corrected $SNR_{co}$')
    axes[1].plot((filtered_cross)/y_cross, df['range'], '.', label='Corrected $SNR_{cross}$')

    axes[0].set_xlim([0.9995, 1.001])
    axes[0].yaxis.set_major_formatter(hd.m_km_ticks())
    axes[0].set_ylabel('Height a.g.l [km]')
    for ax in axes.flatten():
        ax.tick_params(axis='x', labelrotation=45)
        ax.legend()
        ax.set_xlabel('SNR')
    fig.savefig('F:/halo/paper/figures/background_correction2/correction_co_cross' + str(time),
                bbox_inches='tight')

    mask = (avg.aerosol_percentage[time, :] > 0.8) & (avg.range < 2000)
    depo = (cross - 1) / (co - 1)
    depo = depo[mask]
    depo_corrected = ((filtered_cross)/y_cross - 1) / ((filtered)/y_co - 1)
    depo_corrected = depo_corrected[mask]
    total_depo = np.append(depo, total_depo)
    total_depo_corrected = np.append(depo_corrected, total_depo_corrected)

# %%
fig, ax = plt.subplots(figsize=(6, 4), sharex=True, sharey=True)
ax.hist(total_depo[(total_depo < 0.5) & (total_depo > -0.1)], label='Not corrected depo', bins=20)
ax.hist(total_depo_corrected[(total_depo_corrected < 0.5) & (total_depo_corrected > -0.1)],
        alpha=0.5, label='Corrected depo', bins=20)
ax.legend()
ax.set_xlim([-0.01, 0.2])
ax.set_xlabel('$\delta$')
ax.set_ylabel('N')
fig.savefig('F:/halo/paper/figures/background_correction2/summary_myway', bbox_inches='tight')


# %%
time = 3
co = avg['co_signal'][time, :].values
cross = avg['cross_signal_bleed'][time, :].values

coeff = pywt.swt(np.pad(co-1, (0, 67), 'constant', constant_values=(0, 0)), wavelet, level=5)
uthresh = np.median(np.abs(coeff[1]))/0.6745 * np.sqrt(2 * np.log(len(coeff[1])))
coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
filtered = pywt.iswt(coeff, wavelet) + 1
filtered = filtered[:len(co)]

coeff = pywt.swt(np.pad(cross-1, (0, 67), 'constant', constant_values=(0, 0)), wavelet, level=5)
uthresh = np.median(np.abs(coeff[1]))/0.6745 * np.sqrt(2 * np.log(len(coeff[1])))
coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
filtered_cross = pywt.iswt(coeff, wavelet) + 1
filtered_cross = filtered_cross[:len(cross)]

# fig, ax = plt.subplots(2, 1, figsize=(10, 6))
# ax[0].plot(co, label='Raw co')
# ax[0].plot(filtered, label='Filtered co')
# ax[0].legend()
# ax[1].plot(cross, label='Raw cross')
# ax[1].plot(filtered_cross, label='Filtered cross')
# ax[1].legend()
# fig.suptitle(f"SWT Denoising with {wavelet} Wavelet", size=15)
# fig.savefig('F:/halo/paper/figures/background_correction2/denoise_co_cross' + str(time),
#             bbox_inches='tight')

background = filtered < 1+6e-5

fig, ax = plt.subplots()
ax.plot(co, df['range'])
ax.plot(filtered, df['range'])


# selected_range = df['range'][background]
# selected_co = filtered[background]
# selected_cross = filtered_cross[background]
#
# a, b, c = myfit(selected_range, selected_co)
# y_co = c + b*df['range'] + a*(df['range']**2)
# y_co_background = c + b*selected_range + a*(selected_range**2)
#
# a, b, c = myfit(selected_range, selected_cross)
# y_cross = c + b*df['range'] + a*(df['range']**2)
# y_cross_background = c + b*selected_range + a*(selected_range**2)
#
# fig, axes = plt.subplots(1, 3, sharex=True, sharey=True,
#                          figsize=(9, 4))
#
# axes[0].plot(co, df['range'], '.', label='$SNR_{co}$')
# axes[0].plot(cross, df['range'], '.', label='$SNR_{cross}$')
#
# axes[2].plot(filtered, df['range'], '.', label='$SNR_{co}$')
# axes[2].plot(filtered_cross, df['range'], '.', label='$SNR_{cross}$')
#
# axes[0].plot(y_co, df['range'], label='Fitted $SNR_{co}$')
# axes[0].plot(y_cross, df['range'], label='Fitted $SNR_{cross}$')
#
# axes[1].plot((filtered)/y_co, df['range'], '.', label='Corrected $SNR_{co}$')
# axes[1].plot((filtered_cross)/y_cross, df['range'], '.', label='Corrected $SNR_{cross}$')
#
# axes[0].set_xlim([0.9995, 1.001])
# axes[0].yaxis.set_major_formatter(hd.m_km_ticks())
# axes[0].set_ylabel('Height a.g.l [km]')
# for ax in axes.flatten():
#     ax.tick_params(axis='x', labelrotation=45)
#     ax.legend()
#     ax.set_xlabel('SNR')

# %%
time = 1
co = avg['co_signal'][time, :].values
cross = avg['cross_signal_bleed'][time, :].values

coeff = pywt.swt(np.pad(co-1, (0, 67), 'constant', constant_values=(0, 0)), wavelet, level=5)
uthresh = np.median(np.abs(coeff[1]))/0.6745 * np.sqrt(2 * np.log(len(coeff[1])))
coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
filtered = pywt.iswt(coeff, wavelet) + 1
filtered = filtered[:len(co)]

peaks, _ = find_peaks(-filtered)
results = peak_widths(filtered, peaks, rel_height=1)

x = np.arange(len(co))
peaks = peaks[filtered[peaks] < 1]
a, b, c = np.polyfit(peaks, filtered[peaks], deg=2)
smooth_peaks = c + b*x + a*(x**2)
fig, ax = plt.subplots()
ax.plot(co)
ax.plot(filtered)
ax.plot(peaks, filtered[peaks], 'o')
ax.plot(x, smooth_peaks)
ax.plot(x, smooth_peaks + 6e-5)
ax.axhline(y=1+6e-5)
# for i in range(len(results[1])):
#     ax.hlines(results[1][i], results[2][i], results[3][i])
# %%


def background_correction(x, wavelet='bior2.6'):
    coeff = pywt.swt(np.pad(x-1, (0, (len(x) // 2**5 + 3) * 2**5 - len(x)), 'constant', constant_values=(0, 0)),
                     wavelet, level=5)
    uthresh = np.median(np.abs(coeff[1]))/0.6745 * np.sqrt(2 * np.log(len(coeff[1])))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    filtered = pywt.iswt(coeff, wavelet) + 1
    filtered = filtered[:len(x)]

    peaks, _ = find_peaks(-filtered)
    peaks = peaks[filtered[peaks] < 1]

    indices = np.arange(len(x))
    a, b, c = np.polyfit(peaks, filtered[peaks], deg=2)
    smooth_peaks = c + b*indices + a*(indices**2)
    background = filtered < smooth_peaks + 6e-5

    selected_x = filtered[background]
    selected_indices = indices[background]

    a, b, c = np.polyfit(selected_indices, selected_x, deg=2)
    x_fitted = c + b*indices + a*(indices**2)
    x_background_fitted = c + b*selected_indices + a*(selected_indices**2)

    x_corrected = x/x_fitted
    x_background_corrected = selected_x/x_background_fitted

    return x_corrected, x_background_corrected


# %%
corrected_co, _ = background_correction(co)
corrected_cross, _ = background_correction(cross)
aerosol_mask = avg['aerosol_percentage'][time, :] > 0.8

plt.plot(((cross-1) / (co-1))[aerosol_mask], df['range'][aerosol_mask])
plt.plot(((corrected_cross-1) / (corrected_co-1))[aerosol_mask], df['range'][aerosol_mask])

# %%
(len(x) // 2**5 + 3) * 2**5 - len(x)
2**5
len(x) / 2**5
# %%
pywt.swt(np.pad(x-1, (0, 35), 'constant', constant_values=(0, 0)),
         wavelet, level=5)
