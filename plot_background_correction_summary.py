import halo_data as hd
import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import pandas as pd
import xarray as xr
from matplotlib.colors import LogNorm
import string
%matplotlib qt

# %%
sites = ['33', '46', '53', '54']
site_path = r'F:\halo\paper\figures\background_correction_all/stan/' + site + '/'
file_paths = {x: glob.glob(
    r'F:\halo\paper\figures\background_correction_all/stan/' + x + '/' + '**/*.csv') for x in sites}
site_32 = glob.glob(r'F:\halo\paper\figures\background_correction_all/stan/' +
                    '32' + '/' + '**/*.csv')
site_32 = sorted(site_32)
file_paths['32'] = site_32[:513]
file_paths['32XR'] = site_32[513:]

site_plot = ['32', '32XR', '33', '46', '53', '54']

# %%
fig, axes = plt.subplots(6, 2, figsize=(9, 15), sharey='row', sharex='col')
for key, ax in zip(site_plot, axes):
    # for (key, value), ax in zip(file_paths.items(), axes):
    value = file_paths[key]
    df = pd.concat([pd.read_csv(x) for x in value], ignore_index=True)
    df['time'] = pd.to_datetime(df['time'])

    df['subtract'] = df['depo_corrected'] - df['depo']
    temp = df[['co_corrected', 'subtract']]
    temp['co_corrected'] = temp['co_corrected'] - 1
    temp = temp[(temp['subtract'] < 0.3) & (temp['co_corrected'] < 0.2)]
    temp = temp[(temp['subtract'] > -0.1) & (temp['co_corrected'] > 0)]
    x_y_data = temp.dropna()
    H, x_edges, y_edges = np.histogram2d(
        x_y_data['subtract'],
        x_y_data['co_corrected'], bins=500)
    X, Y = np.meshgrid(x_edges, y_edges)
    H[H < 1] = np.nan
    p = ax[0].pcolormesh(X, Y, H.T)
    # ax[0].set_xlabel('$SNR_{co-corrected}$')
    ax[0].set_ylim([0, 0.03])
    colorbar = fig.colorbar(p, ax=ax[0])
    colorbar.ax.set_ylabel('N')
    # line_x = x_edges[:-1]
    # line_y = y_edges[np.argmax(H, axis=1)]
    # line_mask = (line_x < 0.1) & (line_x > 0.001)
    # ax[0].plot(line_x[line_mask], line_y[line_mask], c='red')

    temp = df[['co_corrected', 'depo_corrected_sd']]
    temp['co_corrected'] = temp['co_corrected'] - 1
    temp = temp[(temp['depo_corrected_sd'] < 0.1) & (temp['co_corrected'] < 0.2)]
    temp = temp[(temp['depo_corrected_sd'] > -0.1) & (temp['co_corrected'] > 0)]
    x_y_data = temp.dropna()
    H, x_edges, y_edges = np.histogram2d(
        x_y_data['depo_corrected_sd'],
        x_y_data['co_corrected'],
        bins=500)
    X, Y = np.meshgrid(x_edges, y_edges)
    H[H < 1] = np.nan
    p = ax[1].pcolormesh(X, Y, H.T)
    ax[0].set_ylabel('$SNR_{co,corrected}$')
    ax[1].set_ylim([0, 0.03])
    colorbar = fig.colorbar(p, ax=ax[1])
    colorbar.ax.set_ylabel('N')
    # line_x = x_edges[:-1]
    # line_y = y_edges[np.nanargmax(H, axis=1)]
    # line_mask = (line_x < 0.1) & (line_x > 0.001)
    # ax[1].plot(line_x[line_mask], line_y[line_mask], c='red')

    for ax_ in ax.flatten():
        ax_.grid()
axes[-1, 1].set_xlabel(r'$\sigma_{\delta, corrected}$')
axes[-1, 0].set_xlabel(r'$\delta_{corrected} - \delta_{original}$')

for n, ax_ in enumerate(axes.flatten()):
    ax_.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
             transform=ax_.transAxes, size=12)
fig.savefig(r'F:\halo\paper\figures\background_correction_all/summary.png', bbox_inches='tight')
