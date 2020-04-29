import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns
import matplotlib.ticker as ticker
from pathlib import Path
from matplotlib.colors import LogNorm
%matplotlib qt

# %%
# Define csv directory path
csv_path = r'F:\halo\32\depolarization\depo'
# Create saving folder
depo_result = csv_path + '/result'
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

# %%
depo = pd.melt(depo, id_vars=[x for x in depo.columns if 'depo' not in x],
               value_vars=[x for x in depo.columns if 'depo' in x],
               var_name='depo_type')

# %%
for year in depo.year.unique():
    datelabel = depo[depo['year'] == year].date.unique()
    fig1, ax = plt.subplots(figsize=(18, 9))
    sns.boxplot(
        'date', 'value', hue='depo_type',
        data=depo[(depo['year'] == year) & (depo['value'] < 0.2) & (depo['value'] > 0)],
        ax=ax)
    ax.set_title('Depo at cloud base time series filtered to values in [0, 0.2]',
                 fontweight='bold')
    ax.set_xlabel(year, weight='bold')
    ax.set_ylabel('Depo')
    # ax.set_ylim([0, 0.2])
    # ax.tick_params(axis='x', labelrotation=45)
    # Space out interval for xticks
    ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(0, len(datelabel), 5)))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(
        [x.strftime("%b %d") for x in datelabel][0::5]))
    fig1.savefig(depo_result + '/depo_ts' + str(year) + '.png')

# %%
fig2 = sns.relplot(x='time', y='value',
                   col='month', data=depo[(depo['depo_type'] == 'depo') & (depo['value'] > 0)],
                   alpha=0.2,
                   linewidth=0, col_wrap=3, height=4.5, aspect=4/3)
fig2.fig.savefig(depo_result + '/depo_diurnal.png')

# %%
fig3, ax = plt.subplots(figsize=(18, 9))
depo[(depo['value'] < 0.2) & (depo['value'] > -0.05)].groupby('depo_type')['value'].hist(
    bins=50, ax=ax, alpha=0.5)
ax.legend(['depo', 'depo_1'])
ax.set_title('Distribution of depo at max SNR and 1 level below filtered to values in [-0.05, 0.2]')
ax.set_xlabel('Depo')
fig3.savefig(depo_result + '/depo_hist.png')

# %%
fig4, axes = plt.subplots(3, 2, figsize=(18, 9), sharex=True)
for val, ax in zip(['co_signal', 'co_signal1', 'range',
                    'vraw', 'beta_raw', 'cross_signal'],
                   axes.flatten()):
    ax.plot(depo.groupby('date')[val].mean() if val is not 'beta_raw' else np.log10(
        depo.groupby('date')[val].mean()), '.')
    ax.set_title(val)
fig4.suptitle('Mean values of various metrics at cloud base with max SNR',
              size=22, weight='bold')
fig4.savefig(depo_result + '/depo_other_vars.png')

# %%
fig6, ax = plt.subplots(figsize=(18, 9))
for name, group in depo.groupby('depo_type'):
    ax.plot(group.groupby('date').value.mean(), '.', label=name)
ax.legend()
ax.set_title('Mean value of depo at max SNR and 1 level below')
ax.set_xlabel('Date')
ax.set_ylabel('Depo')
fig6.savefig(depo_result + '/depo_scatter_ts.png')

# %%
fig7 = sns.pairplot(depo_original, vars=['range', 'co_signal', 'cross_signal', 'vraw',
                                         'beta_raw', 'depo'],
                    height=4.5, aspect=4/3)
fig7.savefig(depo_result + '/pairplot.png')

# %%
co_cross_data = depo[['co_signal', 'cross_signal']]
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
fig8.savefig(depo_result + '/cross_vs_co.png')
