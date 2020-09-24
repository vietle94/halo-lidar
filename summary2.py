import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from matplotlib.colors import LogNorm
%matplotlib qt

# %%
save_location = 'F:\\halo\\classifier\\summary\\'
df = pd.read_csv('F:\\halo\\classifier\\32\\result.csv')

df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['depo'][df['depo'] > 1] = np.nan
bin_depo = np.linspace(0, 0.5, 50)
bin_month = np.arange(0.5, 13, 1)
bin_time = np.arange(0, 25)

for key, grp in df.groupby('year'):
    title = str(key) + '_' + df['location'][0]
    H, month_edge, depo_edge = np.histogram2d(
        grp['month'], grp['depo'],
        bins=(bin_month, bin_depo)
    )
    X, Y = np.meshgrid(month_edge, depo_edge)

    fig, ax = plt.subplots(figsize=(12, 6))
    p = ax.pcolormesh(X, Y, H.T, cmap='jet')
    fig.colorbar(p, ax=ax)
    ax.set_xlabel('Month')
    ax.set_ylabel('Depolarization ratio')
    ax.set_title(title, size=22, weight='bold')
    fig.savefig(save_location + title + '_month_depo.png', bbox_inches='tight')

    dep_mean, time_edge, month_edge, _ = binned_statistic_2d(
        grp['time'],
        grp['month'],
        grp['depo'],
        bins=[bin_time, bin_month],
        statistic=np.nanmean)

    X, Y = np.meshgrid(time_edge, month_edge)

    fig, ax = plt.subplots(figsize=(12, 6))
    p = ax.pcolormesh(X, Y, dep_mean.T, cmap='jet', vmin=0, vmax=0.3)
    ax.set_xlabel('Time')
    ax.set_ylabel('Month')
    cbar = fig.colorbar(p, ax=ax)
    cbar.ax.set_ylabel('Depolarization ratio')
    cbar.ax.yaxis.set_label_position('left')
    ax.set_title(title, size=22, weight='bold')
    fig.savefig(save_location + title + '_month_hour.png', bbox_inches='tight')
    plt.close('all')