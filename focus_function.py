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
#######################################


# %%
df = xr.open_dataset(r'F:\halo\classifier_new\32/2019-04-05-Uto-32_classified.nc')
avg = df[['beta_raw', 'co_signal', 'cross_signal']].resample(time='60min').mean(dim='time')

ceilo = Dataset(r'F:\halo\paper\uto_ceilo_20190405\20190405_uto_cl31.nc')
ceilo_beta = ceilo['beta_raw'][:]
ceilo_time = ceilo['time'][:]

ceilo_profile = np.nanmean(ceilo_beta[np.floor(ceilo_time) == 2], axis=0)

# %%


def beta(SNR, R):
    f = 440
    D = 14
    wavelength = df['wavelength'].values[0]
    c = 3e8
    h = 6.626e-34
    E = df['energy'].values[0]
    B = df['bandwidth'].values[0]
    v = 1
    n = 1
    A = np.pi * D**2 / (4*(1+(np.pi*D**2/(4*wavelength*R))**2*(1-R/f)**2))
    T = A/(R)**2
    beta = 2*h*v*B/(n*c*E) * SNR/R

    return beta


fig, ax = plt.subplots(1, 5, sharey=True, figsize=(12, 6))

ax[0].plot(np.log10(ceilo_profile), ceilo['range'][:], '.')
ax[0].set_title(r'$\beta$ ceilometer')

ax[1].plot(np.log10(avg['beta_raw'][0, :]), df['range']/1000, '.')
ax[1].set_title(r'$\beta$ XR')

ax[2].plot(avg['co_signal'][0, :], df['range']/1000, '.')
ax[2].set_title(r'co_signal XR')

ax[3].plot(avg['cross_signal'][0, :], df['range']/1000, '.')
ax[3].set_title(r'cross_signal XR')

ax[4].plot(np.log10(beta(avg['co_signal'][0, :],
                         df['range'])), df['range']/1000, '.')
ax[4].set_title(r'$\beta$ XR')

ax[0].set_ylim([0, 3])
for ax_ in ax.flatten():
    ax_.grid()

ax[0].set_ylabel('Height [km]')
