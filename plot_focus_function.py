from matplotlib.dates import DateFormatter
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
import json
from sklearn.metrics import r2_score
import matplotlib.cm as cm
import calendar
from scipy.stats import binned_statistic_2d
from scipy.stats import binned_statistic
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
import pystan
import pickle
%matplotlib qt

# %%
#######################################

save_dir = r'F:\halo\paper\figures\focus_function/'
dff = xr.open_dataset(r'F:\halo\classifier_new\32/2019-04-05-Uto-32_classified.nc')

df = xr.open_dataset(
    r'F:\halo\32\depolarization\xr/20190405_fmi_halo-doppler-lidar-32-depolarization.nc')
df = df.where(df.range > 90, drop=True)
avg = df[['beta_raw', 'co_signal', 'cross_signal']].resample(time='60min').mean(dim='time')

ceilo = Dataset(r'F:\halo\paper\uto_ceilo_20190405\20190405_uto_cl31.nc')
ceilo_beta = ceilo['beta_raw'][:]
ceilo_range = ceilo['range'][:]
ceilo_time = [pd.Timedelta(x, 'H') + pd.to_datetime(df.time.values[0]).floor('D')
              for x in ceilo['time'][:]]
ceilo_time = pd.to_datetime(ceilo_time)
ceilo_time = ceilo_time - pd.Timedelta('2H')

ceilo_beta = ceilo_beta[ceilo_time > pd.to_datetime('20190405')]
ceilo_time = ceilo_time[ceilo_time > pd.to_datetime('20190405')]

ceilo_time_seconds = ceilo_time.hour*3600+ceilo_time.minute*60+ceilo_time.second
halo_time = pd.to_datetime(df.time.values)
halo_time_seconds = halo_time.hour*3600+halo_time.minute*60+halo_time.second
ceilo_profile = np.nanmean(ceilo_beta[ceilo_time.hour < 1], axis=0)

# %%


def beta(SNR, R, f=308.399, D=18.565e-3):
    wavelength = df['wavelength'].values  # 1.5e-6 m
    c = 3e8
    h = 6.626e-34
    E = df['energy'].values  # 9.99e-6 J
    B = df['bandwidth'].values  # 50000000 Hz
    v = c/wavelength
    n = 1
    A = np.pi * (D**2) / (4*(1+(np.pi*(D**2)/(4*wavelength*R))**2*(1-R/f)**2))
    T = A/(R**2)
    beta = (2*h*v*B)/(n*c*E) * (SNR/T)

    return beta

#
# # %%
# fig, ax = plt.subplots(3, 3, sharey=True, sharex=True, figsize=(12, 6))
#
# for ax_, (f, D) in zip(ax.flatten(), list(itertools.product([400, 500, 600], [12e-3, 17e-3, 22e-3]))):
#     ax_.plot(np.log10(beta(avg['co_signal'][0, :]-1,
#                            df['range'], f, D)), df['range']/1000, '.')
#     ax_.set_title(f'f:{f}m, D:{D}m')
#
#
# # ax[0].set_ylim([0, 3])
# for ax_ in ax.flatten():
#     ax_.grid()
#     ax_.set_xlim([-8, -4])

# %%


df['co_signal'] = xr.where(
    df['co_signal'] < 3*dff.attrs['background_snr_sd'] + 1, np.nan, df['co_signal'])
halo_range = df['range'].values
min_range = 200 + 16
halo_range = halo_range[halo_range > min_range]
final_range_bin = np.append((halo_range - 15), (halo_range + 15)[-1])
final_range_bin = final_range_bin[final_range_bin < 7710]
halo_snr = (df['co_signal']-1)[:, df['range'] > min_range]
final_time_bin = pd.date_range('2019-04-05', periods=34, freq='30min')
final_time_bin_seconds = final_time_bin.hour*3600 + final_time_bin.minute*60 + final_time_bin.minute

# %%
ceilo_beta_flattened = ceilo_beta.flatten()
ceilo_time_seconds_flattened = np.repeat(ceilo_time_seconds, ceilo_range.size)
ceilo_range_flattened = np.tile(ceilo_range*1000, ceilo_time_seconds.size)
ceilo_beta_binned, _, _, _ = binned_statistic_2d(ceilo_time_seconds_flattened,
                                                 ceilo_range_flattened, ceilo_beta_flattened,
                                                 bins=[final_time_bin_seconds, final_range_bin],
                                                 statistic=np.nanmean)

# %%
halo_snr_flattened = halo_snr.values.flatten()
halo_time_seconds_flattened = np.repeat(halo_time_seconds, halo_range.size)
halo_range_flattened = np.tile(halo_range, halo_time_seconds.size)
halo_snr_binned, _, _, _ = binned_statistic_2d(halo_time_seconds_flattened,
                                               halo_range_flattened, halo_snr_flattened,
                                               bins=[final_time_bin_seconds, final_range_bin],
                                               statistic=np.nanmean)

# %%
data = pd.DataFrame({'halo_snr': halo_snr_binned.flatten(),
                     'ceilo_beta': ceilo_beta_binned.flatten(),
                     'time': np.repeat(final_time_bin[:-1], final_range_bin.size-1),
                     'range': np.tile(final_range_bin[:-1]+15, final_time_bin.size - 1)})


# %%
data['ceilo_beta'] = np.log10(data['ceilo_beta'])
data = data.dropna(axis=0)
data = data.reset_index(drop=True)
data = data[data['range'] < 1900]

# %%
# f = np.linspace(100, 1000, 100)
# D = np.linspace(10e-3, 60e-3, 100)
#
# result_rms = []
# result_f = []
# result_D = []
# for grp_name, grp_value in data.groupby(data['time']):
#     rms_base = 1100
#     f_base = 900
#     D_base = 900
#     for fi, f_ in enumerate(f):
#         for Di, D_ in enumerate(D):
#             corrected_beta = beta(grp_value['halo_snr'], grp_value['range'], f=f_, D=D_)
#             corrected_beta = np.log10(corrected_beta)
#             new_rms = np.sqrt(np.mean((data['ceilo_beta'] - corrected_beta)**2))
#             if new_rms < rms_base:
#                 rms_base = new_rms
#                 f_base = f_
#                 D_base = D_
#     result_rms.append(rms_base)
#     result_f.append(f_base)
#     result_D.append(D_base)

# %%
# %matplotlib qt
# fig, ax = plt.subplots(figsize=(12, 9))
# H, x_edges, y_edges = np.histogram2d(
#     result_f,
#     result_D,
#     bins=[np.linspace(0, 1000, 120), np.linspace(0, 60e-3, 120)])
#
# H[H == 0] = np.nan
# X, Y = np.meshgrid(x_edges, y_edges)
# p = ax.pcolormesh(x_edges, y_edges, H.T)
# ax.set_xlim([000, 1000])
# ax.set_ylim([0e-3, 60e-3])
# ax.set_xlabel('f')
# ax.set_ylabel('D')
# cbar = fig.colorbar(p, ax=ax)
# cbar.ax.set_ylabel('N')
# fig.savefig(save_dir + '2d_histogram_f_D', bbox_inches='tight')

# %%
# scaler = StandardScaler()
# r2_matrix = np.empty((f.size, D.size))
# rms_matrix = np.empty((f.size, D.size))
#
# for fi, f_ in enumerate(f):
#     for Di, D_ in enumerate(D):
#         corrected_beta = beta(data['halo_snr'], data['range'], f=f_, D=D_)
#         corrected_beta = np.log10(corrected_beta)
#         corrected_beta_compare = corrected_beta
#         ceilo_beta_compare = data['ceilo_beta']
#         r2_matrix[fi, Di] = pearsonr(ceilo_beta_compare, corrected_beta_compare)[0]
#         rms_matrix[fi, Di] = np.sqrt(np.mean((corrected_beta_compare - ceilo_beta_compare)**2))
#
#
# # %%
# fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True, sharey=True)
# p = ax[0].pcolormesh(f, D, r2_matrix.T)
# fig.colorbar(p, ax=ax[0])
# ax[0].set_title('Pearson_R')
#
# pp = ax[1].pcolormesh(f, D, rms_matrix.T)
# fig.colorbar(pp, ax=ax[1])
# ax[1].set_title('root_mean_squared_error')
#
# # %%
# fig, ax = plt.subplots(3, 3, sharey=True, sharex=True, figsize=(12, 6))
#
# for ax_, (f, D) in zip(ax.flatten(), list(itertools.product([100, 300, 900], [10e-3, 18e-3, 30e-3]))):
#     mask = data['range'] < 1100
#     x_data = np.log10(beta(data['halo_snr'][mask], data['range'][mask], f, D))
#     y_data = data['ceilo_beta'][mask]
#     ax_.plot(x_data, y_data, '.', alpha=0.1)
#     ax_.set_title(f'f:{f}m, D:{D}m')
#
#     axline_min = np.nanmin((np.nanmin(x_data), np.nanmin(y_data)))
#     axline_max = np.nanmin((np.nanmax(x_data), np.nanmax(y_data)))
#     ax_.axline((axline_min, axline_min),
#                (axline_max, axline_max),
#                color='grey', linewidth=0.5, ls='--')
#
#     ax_.grid()
#     ax_.set_xlim([-8, -4])
#     ax_.set_ylim([-8, -4])
#     ax_.set_xticks([-8, -7, -6, -5, -4])


# %%
fig, ax = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(7, 4))

ax[0].plot(ceilo_profile, ceilo['range'][:], '.')

ax[1].plot(avg['beta_raw'][0, :], df['range']/1000, '.')

ax[2].plot(beta(avg['co_signal'][0, :]-1,
                df['range']), df['range']/1000, '.')

for ax_ in ax.flatten():
    ax_.grid()
    ax_.set_xscale('log')
    ax_.set_xlabel(r'$\beta \quad[Mm^{-1}]$')
    ax_.set_xlim(left=10e-8)
    ax_.set_ylim(top=8)

ax[0].set_ylabel('Height a.g.l [km]')

for n, ax_ in enumerate(ax.flatten()):
    ax_.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
             transform=ax_.transAxes, size=12)
fig.subplots_adjust(bottom=0.2)
fig.savefig(save_dir + 'profile.png', bbox_inches='tight')

# %%

fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(9, 6))
p2 = ax[2].pcolormesh(df.time.values, df.range,
                      np.log10(beta(df['co_signal']-1,
                                    df['range'])).T, cmap='jet',
                      vmin=-8, vmax=-4)
p1 = ax[1].pcolormesh(df.time.values, df.range, np.log10(df.beta_raw).T, cmap='jet',
                      vmin=-8, vmax=-4)
p0 = ax[0].pcolormesh(ceilo_time, ceilo_range*1000, np.log10(ceilo_beta).T, cmap='jet',
                      vmin=-8, vmax=-4)

for ax_, p, lab in zip(ax.flatten(), [p0, p1, p2],
                       [r'$\beta \quad[Mm^{-1}]$',
                        r'$\beta \quad[Mm^{-1}]$',
                        r'$\beta \quad[Mm^{-1}]$']):
    ax_.yaxis.set_major_formatter(hd.m_km_ticks())
    ax_.set_yticks([0, 4000, 8000])
    cbar = fig.colorbar(p, ax=ax_, fraction=0.05)
    cbar.ax.set_ylabel(lab, rotation=90)
    cbar.ax.set_title(r'$1e$', size=10)
    ax_.set_ylabel('Height a.g.l [km]')
myFmt = DateFormatter("%H")
ax[2].set_xlim(left=pd.to_datetime(df.time[0].values).floor('D'))
ax[2].xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
ax[2].xaxis.set_major_formatter(myFmt)
ax[2].set_xlim(left=pd.to_datetime(df.time[0].values).floor('D'))
ax[2].set_xlabel('Time UTC [hour]')

for n, ax_ in enumerate(ax.flatten()):
    ax_.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
             transform=ax_.transAxes, size=12)
fig.savefig(save_dir + "focus_function_result.png", bbox_inches='tight')

# %%
