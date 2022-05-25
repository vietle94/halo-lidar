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
import itertools
%matplotlib qt

# %%
#######################################


# %%
df = xr.open_dataset(r'F:\halo\classifier_new\32/2019-04-05-Uto-32_classified.nc')
avg = df[['beta_raw', 'co_signal', 'cross_signal']].resample(time='60min').mean(dim='time')

ceilo = Dataset(r'F:\halo\paper\uto_ceilo_20190405\20190405_uto_cl31.nc')
ceilo_beta = ceilo['beta_raw'][:]
ceilo_time = ceilo['time'][:]
ceilo_range = ceilo['range'][:]

ceilo_profile = np.nanmean(ceilo_beta[np.floor(ceilo_time) == 2], axis=0)

# %%


def beta(SNR, R, f=625, D=12e-3):
    # f = 625
    # D = 12e-3
    wavelength = df['wavelength'].values[0]  # 1.5e-6 m
    c = 3e8
    h = 6.626e-34
    E = df['energy'].values[0]  # 9.99e-6 J
    B = df['bandwidth'].values[0]  # 50000000 Hz
    v = c/wavelength
    n = 1
    A = np.pi * (D**2) / (4*(1+(np.pi*(D**2)/(4*wavelength*R))**2*(1-R/f)**2))
    T = A/(R**2)
    beta = (2*h*v*B)/(n*c*E) * (SNR/T)

    return beta


fig, ax = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(12, 6))

ax[0].plot(np.log10(ceilo_profile), ceilo['range'][:], '.')
ax[0].set_title(r'$\beta$ ceilometer')

ax[1].plot(np.log10(avg['beta_raw'][0, :]), df['range']/1000, '.')
ax[1].set_title(r'$\beta$ XR')

# ax[2].plot(avg['co_signal'][0, :], df['range']/1000, '.')
# ax[2].set_title(r'co_signal XR')
#
# ax[3].plot(avg['cross_signal'][0, :], df['range']/1000, '.')
# ax[3].set_title(r'cross_signal XR')

ax[2].plot(np.log10(beta(avg['co_signal'][0, :]-1,
                         df['range'])), df['range']/1000, '.')
ax[2].set_title(r'$\beta$ XR - corrected')

# ax[0].set_ylim([0, 3])
for ax_ in ax.flatten():
    ax_.grid()

ax[0].set_ylabel('Height [km]')

# %%
fig, ax = plt.subplots(3, 3, sharey=True, sharex=True, figsize=(12, 6))

for ax_, (f, D) in zip(ax.flatten(), list(itertools.product([400, 500, 600], [12e-3, 17e-3, 22e-3]))):
    ax_.plot(np.log10(beta(avg['co_signal'][0, :]-1,
                           df['range'], f, D)), df['range']/1000, '.')
    ax_.set_title(f'f:{f}m, D:{D}m')


# ax[0].set_ylim([0, 3])
for ax_ in ax.flatten():
    ax_.grid()
    ax_.set_xlim([-8, -4])


# %%
list(itertools.product([400, 500, 600], [12, 17, 22]))

# %%
avg['co_signal'].shape
beta_corrected = np.array([beta(df['co_signal'][x, :].values-1,
                                df['range'].values) for x in range(df['co_signal'].shape[0])])

# %%
beta_corrected.shape

# %%

fig, ax = plt.subplots(3, 1, sharex=True, sharey=True)
p0 = ax[0].pcolormesh(df.time, df.range, np.log10(beta_corrected).T, cmap='jet',
                      vmin=-8, vmax=-4)
p1 = ax[1].pcolormesh(df.time, df.range, np.log10(df.beta_raw).T, cmap='jet',
                      vmin=-8, vmax=-4)
p2 = ax[2].pcolormesh(ceilo_time, ceilo['range'][:]*1000, np.log10(ceilo_beta).T, cmap='jet',
                      vmin=-8, vmax=-4)

for ax_, p, lab in zip(ax.flatten(), [p0, p1, p2],
                       [r'corrected $\beta$', r'original $\beta$',
                        'from ceilometer']):
    ax_.yaxis.set_major_formatter(hd.m_km_ticks())
    ax_.set_yticks([0, 4000, 8000])
    cbar = fig.colorbar(p, ax=ax_, fraction=0.05)
    cbar.ax.set_ylabel(lab, rotation=90)
    cbar.ax.set_title(r'$1e$', size=10)
    ax_.set_ylabel('Height a.g.l [km]')
# %%
