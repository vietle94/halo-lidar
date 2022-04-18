import halo_data as hd
import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import pandas as pd
import xarray as xr
from matplotlib.colors import LogNorm

%matplotlib qt

# %%
sites = ['32', '33', '46', '53']
for site in sites:
    site_path = r'F:\halo\paper\figures\background_correction_all/' + site + '/'
    file_paths = glob.glob(site_path + '**/*.csv')
    df = pd.concat([pd.read_csv(x) for x in file_paths], ignore_index=True)
    df['time'] = pd.to_datetime(df['time'])

    fig, ax = plt.subplots(1, 2, figsize=(16, 9))
    good_percentage = np.sum(df['good_profile'])/len(df['good_profile'])
    sizes = [good_percentage, 1 - good_percentage]
    sizes_label = ['Good data points', 'Bad data points']
    ax[0].pie(sizes, labels=sizes_label, startangle=90, autopct='%1.1f%%')

    good_profile_first = df.groupby([df['time'].dt.date, df['time'].dt.hour])[
        'good_profile'].first()
    good_profile_mean = good_profile_first.mean()
    sizes2 = [good_profile_mean, 1 - good_profile_mean]
    sizes2_label = ['Good profiles', 'Bad profiles']
    ax[1].pie(sizes2, labels=sizes2_label, startangle=90, autopct='%1.1f%%')
    fig.savefig(r'F:\halo\paper\figures\background_correction_all/' + site +
                '_good_profile.png', bbox_inches=None)

    df['subtract'] = df['depo_wave'] - df['depo']
    temp = df[['co_corrected', 'subtract']]
    temp['co_corrected'] = temp['co_corrected'] - 1
    temp = temp[(temp['subtract'] < 0.3) & (temp['co_corrected'] < 0.2)]
    temp = temp[(temp['subtract'] > -0.1) & (temp['co_corrected'] > 0)]
    x_y_data = temp.dropna()
    H, x_edges, y_edges = np.histogram2d(
        x_y_data['co_corrected'],
        x_y_data['subtract'],
        bins=1000)
    X, Y = np.meshgrid(x_edges, y_edges)
    fig, ax = plt.subplots(figsize=(16, 9))
    p = ax.pcolormesh(X, Y, H.T)
    ax.set_xlabel('$SNR_{co-corrected}$')
    ax.set_ylabel('Depo_corrected - depo_original')
    ax.set_xlim([0, 0.03])
    colorbar = fig.colorbar(p, ax=ax)
    colorbar.ax.set_ylabel('N')
    ax.grid()
    line_x = x_edges[:-1]
    line_y = y_edges[np.argmax(H, axis=1)]
    line_mask = (line_x < 0.1) & (line_x > 0.001)
    ax.plot(line_x[line_mask], line_y[line_mask], c='red')
    fig.savefig(r'F:\halo\paper\figures\background_correction_all/' + site +
                '_co_cutoff.png', bbox_inches=None)
    plt.close('all')

# %%
