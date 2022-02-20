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
data = hd.getdata('F:/halo/32/depolarization/xr')
date = '20190405'
file = [file for file in data if date in file][0]
df = hd.halo_data(file)

df.filter_height()
df.unmask999()
df.depo_cross_adj()

with open('ref_XR2.npy', 'rb') as f:
    ref = np.load(f)
df.filter(variables=['beta_raw'],
          ref='co_signal',
          threshold=1 + 3*df.snr_sd)

log_beta = np.log10(df.data['beta_raw'])
log_beta2 = log_beta.copy()
log_beta2[:, :50] = log_beta2[:, :50] - ref


fig, ax = plt.subplots(3, 1, figsize=(6, 6), sharex=True)
for ax_, beta in zip(ax.flatten()[:-1], [log_beta, log_beta2]):
    p = ax_.pcolormesh(df.data['time'], df.data['range'],
                       beta.T, cmap='jet', vmin=-8, vmax=-4)
    cbar = fig.colorbar(p, ax=ax_)
    cbar.ax.set_ylabel(r'$\beta\quad[Mm^{-1}]$')
    cbar.ax.set_title(r'$1e$', size=10)
    ax_.yaxis.set_major_formatter(hd.m_km_ticks())
    ax_.set_ylabel('Height a.g.l [km]')
    ax_.set_xticks([0, 6, 12, 18, 24])
    ax_.set_yticks([0, 4000, 8000, 12000])

ceilo = Dataset(r'F:\halo\paper\uto_ceilo_20190405\20190405_uto_cl31.nc')
ax[-1].pcolormesh(ceilo['time'][:], ceilo['range'][:], np.log10(ceilo['beta'][:]).T,
                  vmin=-8, vmax=-4, cmap='jet')
cbar = fig.colorbar(p, ax=ax[-1])
cbar.ax.set_ylabel(r'$\beta\quad[Mm^{-1}]$')
cbar.ax.set_title(r'$1e$', size=10)
ax[-1].set_ylabel('Height a.g.l [km]')
ax[-1].set_xticks([0, 6, 12, 18, 24])
ax[-1].set_yticks([0, 4, 8, 12])
ax[-1].set_xlabel('Time UTC [hour]')

for n, ax_ in enumerate(ax.flatten()):
    ax_.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
             transform=ax_.transAxes, size=12)
fig.tight_layout()
# fig.savefig('F:/halo/paper/figures/XR_correction_' + df.filename +
#             '.png', bbox_inches='tight')

# %%


def beta_averaged(x):
    '''
    Calculate beta from co_signal_averaged, which take into account focus
    '''
    c_light = 299792458
    hnu = 1.28e-19
    alpha = 0.01

    x.info['lens_diameter'] = x.info.get('lens_diameter', 0.06)
    x.info['wavelength'] = x.info.get('wavelength', 1.5e-6)
    x.info['beam_energy'] = x.info.get('beam_energy', 0.00001)

    het_area = np.pi * (0.7 * x.info['lens_diameter'] / 2) ** 2

    if x.info['focus'] < 0:
        nt = het_area / x.info['wavelength'] / x.data['range']
    else:
        nt = het_area / x.info['wavelength'] * (1 / x.info['focus'] -
                                                1 / x.data['range'])

    effective_area = het_area / (1 + nt ** 2)
    T = 10 ** (-1 * alpha * x.data['range'] / 5000)
    het_cnr = 0.7 * 0.4 * 0.6 * effective_area * c_light * \
        x.info['beam_energy'] * T / (2 * x.data['range'] ** 2 *
                                     hnu * x.info['bandwidth'])

    pr2 = (x.data['co_signal'] - 1) / het_cnr.T
    return pr2


# %%
temp = beta_averaged(df)
fig, ax = plt.subplots()
ax.pcolormesh(np.log10(temp).T, vmin=-8, vmax=-4, cmap='jet')

# %%


def beta(x):
    '''
    Calculate beta from co_signal_averaged, which take into account focus
    '''
    c_light = 299792458
    hnu = 1.28e-19
    alpha = 0.01

    x.info['lens_diameter'] = x.info.get('lens_diameter', 0.06)
    x.info['wavelength'] = x.info.get('wavelength', 1.5e-6)
    x.info['beam_energy'] = x.info.get('beam_energy', 0.00001)

    het_area = np.pi * (0.7 * x.info['lens_diameter'] / 2) ** 2

    nt = het_area / x.info['wavelength'] / x.data['range']

    effective_area = het_area / (1 + nt ** 2)
    T = 10 ** (-1 * alpha * x.data['range'] / 5000)
    het_cnr = 0.7 * 0.4 * 0.6 * effective_area * c_light * \
        x.info['beam_energy'] * T / (2 * x.data['range'] ** 2 *
                                     hnu * x.info['bandwidth'])

    pr2 = (x.data['co_signal'] - 1) / het_cnr.T
    return pr2

# %%


def detect_2base(x, lim):
    for i, _ in enumerate(x):
        if (x[i] < lim) & (x[i+1] > lim):
            return i


# %%
wavelet = 'bior2.6'
df = xr.open_dataset(r'F:\halo\classifier_new\32/2019-04-05-Uto-32_classified.nc')
avg = df['co_signal'].resample(time='30min').mean(dim='time')
for time in range(24):
    co = avg[time, :].values

    coeff = pywt.swt(np.pad(co-1, (0, 51), 'constant', constant_values=(0, 0)),
                     wavelet, level=5)
    uthresh = np.median(np.abs(coeff[1]))/0.6745 * np.sqrt(2 * np.log(len(coeff[1])))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    filtered = pywt.iswt(coeff, wavelet) + 1
    filtered = filtered[:len(co)]

    cut_off = detect_2base(filtered, 1 + 6e-5) * 30

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['range'], co, label='Raw')
    ax.plot(df['range'], filtered, label='Filtered')
    ax.axhline(y=1+6e-5)
    ax.set_ylim([-0.0004+1, 0.0106+1])
    ax.axvline(x=cut_off)

# %%
co = avg[time, :].values

coeff = pywt.swt(np.pad(co-1, (0, 51), 'constant', constant_values=(0, 0)),
                 wavelet, level=5)
uthresh = np.median(np.abs(coeff[1]))/0.6745 * np.sqrt(2 * np.log(len(coeff[1])))
coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
filtered = pywt.iswt(coeff, wavelet) + 1
filtered = filtered[:len(co)]

cut_off = detect_2base(temp, 1 + 6e-5) * 30

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df['range'], co, label='Raw')
ax.plot(df['range'], filtered, label='Filtered')
ax.axhline(y=1+6e-5)
ax.set_ylim([-0.0004+1, 0.0106+1])
ax.axvline(x=cut_off)
