import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns
import matplotlib.ticker as ticker
from pathlib import Path
from matplotlib.colors import LogNorm

# %%
# Define csv directory path
csv_path = r'F:\halo\32\depolarization\depo'
# Create saving folder
depo_result = csv_path + '/result_old_XR'
Path(depo_result).mkdir(parents=True, exist_ok=True)
# Collect csv file in csv directory and subdirectory
data_list = all_csv_files = [file
                             for path, subdir, files in os.walk(csv_path)
                             for file in glob.glob(os.path.join(path, '*.csv'))]

# %%
depo = pd.concat([pd.read_csv(f) for f in data_list],
                 ignore_index=True)
depo = depo.astype({'year': int, 'month': int, 'day': int})

# For right now, just take the date, ignore hh:mm:ss
depo['date'] = pd.to_datetime(depo[['year', 'month', 'day']]).dt.date
depo_original = depo
old_original = depo_original[depo_original['date'] < pd.to_datetime('2017-11-22')]
xr_original = depo_original[depo_original['date'] >= pd.to_datetime('2017-11-22')]
# %%
depo = pd.melt(depo, id_vars=[x for x in depo.columns if 'depo' not in x],
               value_vars=[x for x in depo.columns if 'depo' in x],
               var_name='depo_type')

old = depo[depo['date'] < pd.to_datetime('2017-11-22')]
xr = depo[depo['date'] >= pd.to_datetime('2017-11-22')]


# %%
for device, label in zip([old, xr], ['old', 'xr']):
    fig2 = sns.relplot(x='time', y='value',
                       col='month',
                       data=device[(device['depo_type'] == 'depo')],
                       alpha=0.2,
                       linewidth=0, col_wrap=3, height=4.5, aspect=4/3)
    fig2.fig.savefig(depo_result + '/' + label + '_diurnal.png')

# %%
fig3, axes = plt.subplots(2, 1, figsize=(18, 9), sharex=True)
for device, label, ax in zip([old, xr], ['old', 'xr'], axes.flatten()):
    device[(device['value'] < 0.2) & (device['value'] > -0.05)].groupby('depo_type')['value'].hist(
        bins=50, ax=ax, alpha=0.5)
    ax.legend(['depo', 'depo_1'])
    ax.set_title(
        'Distribution of depo at max SNR and 1 level below filtered to values of ' + label + ' device')
    ax.set_xlabel('Depo')
fig3.savefig(depo_result + '/depo_hist.png',
             dpi=200, bbox_inches='tight')

# %%
for device, label in zip([old, xr], ['old', 'xr']):
    fig4, axes = plt.subplots(3, 2, figsize=(18, 9), sharex=True)
    for val, ax in zip(['co_signal', 'co_signal1', 'range',
                        'vraw', 'beta_raw', 'cross_signal'],
                       axes.flatten()):
        ax.plot(device.groupby('date')[val].mean() if val is not 'beta_raw' else np.log10(
            device.groupby('date')[val].mean()), '.')
        ax.set_title(val)
        ax.tick_params(axis='x', rotation=45)
    fig4.suptitle('Mean values of various metrics at cloud base with max SNR of ' + label + ' device',
                  size=22, weight='bold')
    fig4.savefig(depo_result + '/' + label + '_other_vars.png',
                 dpi=200, bbox_inches='tight')


# %%
for device, label in zip([old_original, xr_original], ['old', 'xr']):
    fig7 = sns.pairplot(device, vars=['range', 'co_signal', 'cross_signal',
                                      'vraw', 'beta_raw', 'depo'],
                        height=4.5, aspect=4/3)
    fig7.savefig(depo_result + '/' + label + '_pairplot.png')

# %%
co_cross_data = old[['co_signal', 'cross_signal']]
co_cross_data.dropna(inplace=True)
H, co_edges, cross_edges = np.histogram2d(co_cross_data['co_signal'] - 1,
                                          co_cross_data['cross_signal'] - 1,
                                          bins=500)
X, Y = np.meshgrid(co_edges, cross_edges)
fig8, ax = plt.subplots(figsize=(18, 9))
p = ax.pcolormesh(X, Y, H.T, norm=LogNorm())
ax.set_xlabel('co_signal - 1')
ax.set_ylabel('cross_signal - 1')
colorbar = fig8.colorbar(p, ax=ax)
colorbar.ax.set_ylabel('Number of observations')
colorbar.ax.yaxis.set_label_position('left')
ax.set_title('2D histogram of cross_signal vs co_signal', size=22, weight='bold')
ax.plot(co_cross_data['co_signal'] - 1,
        (co_cross_data['co_signal'] - 1) * 0.01, label='depo 0.01 fit',
        linewidth=0.5)
ax.plot(co_cross_data['co_signal'] - 1,
        (co_cross_data['co_signal'] - 1) * 0.07, label='depo 0.07 fit',
        linewidth=0.5)
ax.legend(loc='upper left')
fig8.savefig(depo_result + '/old_cross_vs_co.png',
             dpi=200, bbox_inches='tight')

# %%

co_cross_data = xr[['co_signal', 'cross_signal']]
co_cross_data.dropna(inplace=True)
H, co_edges, cross_edges = np.histogram2d(co_cross_data['co_signal'] - 1,
                                          co_cross_data['cross_signal'] - 1,
                                          bins=500)
X, Y = np.meshgrid(co_edges, cross_edges)
fig9, ax = plt.subplots(figsize=(18, 9))
p = ax.pcolormesh(X, Y, H.T, norm=LogNorm())
ax.set_xlabel('co_signal - 1')
ax.set_ylabel('cross_signal - 1')
colorbar = fig9.colorbar(p, ax=ax)
colorbar.ax.set_ylabel('Number of observations')
colorbar.ax.yaxis.set_label_position('left')
ax.set_title('2D histogram of cross_signal vs co_signal', size=22, weight='bold')
ax.plot(co_cross_data['co_signal'] - 1,
        (co_cross_data['co_signal'] - 1) * 0.01, label='depo 0.01 fit',
        linewidth=0.5)
ax.plot(co_cross_data['co_signal'] - 1,
        (co_cross_data['co_signal'] - 1) * 0.07, label='depo 0.07 fit',
        linewidth=0.5)
ax.legend(loc='upper left')
fig9.savefig(depo_result + '/xr_cross_vs_co.png',
             dpi=200, bbox_inches='tight')
