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
csv_path = r'F:\halo\146\depolarization\depo_saturation'
# Create saving folder
depo_result = csv_path + '/result'
Path(depo_result).mkdir(parents=True, exist_ok=True)
# Collect csv file in csv directory and subdirectory
data_list = [file
             for path, subdir, files in os.walk(csv_path)
             for file in glob.glob(os.path.join(path, '*.csv'))]

# %%
depo = pd.read_csv(data_list[0])
depo = depo.astype({'year': int, 'month': int, 'day': int})
# For right now, just take the date, ignore hh:mm:ss
depo['hour'] = depo['time'].astype(int)
depo['minute'] = (depo['time'] * 60) % 60
depo['second'] = (depo['time'] * 3600) % 60
depo['datetime'] = pd.to_datetime(depo[['year', 'month', 'day',
                                        'hour', 'minute', 'second']])
# depo.set_index('datetime', inplace=True)
depo.drop(['year', 'month', 'day',
           'hour', 'minute', 'second'], axis=1, inplace=True)

# %%
# max_snr = depo.sort_values(
#     'co_signal', ascending=True).drop_duplicates('datetime', keep='last')
#
# max_unsaturated_snr = depo.loc[depo['co_signal'] < 3, :].sort_values(
#     'co_signal', ascending=True).drop_duplicates('datetime', keep='last')

# %%
# depo_saturate = depo[depo['co_signal'] < 3]
# mask = depo_saturate.groupby('datetime')['co_signal'].transform(
#     lambda x: x > (x.max() * 0.95))
# depo_aerosol_saturate = depo_saturate.loc[mask, :]

# %%
fig, ax = plt.subplots(3, 2, figsize=(12, 9))
for ax, val in zip(ax.flatten(), [0.05, 0.1, 0.3, 0.5, 0.7, 0.95]):
    depo_saturate = depo[depo['co_signal'] < 3]
    mask = depo_saturate.groupby('datetime')['co_signal'].transform(
        lambda x: (x-1) > ((x.max()-1) * val))
    depo_aerosol_saturate = depo_saturate.loc[mask, :]
    depo_aerosol_saturate_min = depo_aerosol_saturate.groupby('datetime')['depo'].min()
    range_mask = (depo_aerosol_saturate_min < 0.5) & (depo_aerosol_saturate_min > -0.1)
    ax.hist(depo_aerosol_saturate_min[range_mask], bins=50)
    ax.set_title(val)
fig.subplots_adjust(hspace=0.3)
fig.suptitle(
    'Histogram of depo at different snr threshold (relative to max snr) for aerosol \n and co_signal<3 for saturation',
    weight='bold')

# %%
fig, ax = plt.subplots(figsize=(12, 9))
ax.hist(depo[depo['depo'] > -0.1].groupby('datetime')['depo'].min(), bins=50)
ax.set_title('Histogram of minimum depo from each profile',
             weight='bold')

# %%
fig, ax = plt.subplots(figsize=(12, 9))
ax.hist(depo.loc[(depo['type'] == 0) & (depo['depo'] < 0.2) & (depo['depo'] > -0.05), 'depo'],
        bins=50)
# %%
fig, ax = plt.subplots(2, 1, figsize=(12, 9))
ax[0].hist(max_snr.loc[max_snr['depo'] < 0.5, 'depo'], bins=20)
ax[0].set_title('Depo at max SNR')
ax[1].hist(max_unsaturated_snr.loc[max_unsaturated_snr['depo'] < 0.5, 'depo'],
           bins=20)
ax[1].set_title('Depo at max SNR with co_signal-1 < 2 (Controlled saturation)')

# %%
fig, ax = plt.subplots(figsize=(12, 9))
max_unsaturated_snr['type'].hist(ax=ax)
ax.set_title('Level below max SNR which depo was selected (Controlled saturation)')

# %%
fig, ax = plt.subplots(2, 1, figsize=(12, 9))
ax[0].hist(max_snr.loc[max_snr['depo'] < 0.5, 'co_signal'], bins=20)
ax[0].set_title('Distribution of co_signal at max SNR')
ax[1].hist(max_unsaturated_snr.loc[max_unsaturated_snr['depo'] < 0.5, 'co_signal'],
           bins=20)
ax[1].set_title('Distribution of co_signal at max SNR with co_signal-1 < 2 (Controlled saturation)')

# %%
fig, ax = plt.subplots(figsize=(12, 9))
ax.hist(max_snr.loc[(max_snr['depo'] < 0.5) & (max_snr['co_signal'] < 3), 'depo'], bins=50)

#
# # %%
# depo = pd.melt(depo, id_vars=[x for x in depo.columns if 'depo' not in x],
#                value_vars=[x for x in depo.columns if 'depo' in x],
#                var_name='depo_type')
#
# # %%
# for year in depo.year.unique():
#     datelabel = depo[depo['year'] == year].date.unique()
#     fig1, ax = plt.subplots(figsize=(18, 9))
#     sns.boxplot(
#         'date', 'value', hue='depo_type',
#         data=depo[(depo['year'] == year) & (depo['value'] < 0.2) & (depo['value'] > 0)],
#         ax=ax)
#     ax.set_title('Depo at cloud base time series filtered to values in [0, 0.2]',
#                  fontweight='bold')
#     ax.set_xlabel(year, weight='bold')
#     ax.set_ylabel('Depo')
#     # ax.set_ylim([0, 0.2])
#     # ax.tick_params(axis='x', labelrotation=45)
#     # Space out interval for xticks
#     ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(0, len(datelabel), 5)))
#     ax.xaxis.set_major_formatter(ticker.FixedFormatter(
#         [x.strftime("%b %d") for x in datelabel][0::5]))
#     fig1.savefig(depo_result + '/depo_ts' + str(year) + '.png')
#
# # %%
# fig2 = sns.relplot(x='time', y='value',
#                    col='month', data=depo[(depo['depo_type'] == 'depo') & (depo['value'] > 0)],
#                    alpha=0.2,
#                    linewidth=0, col_wrap=3, height=4.5, aspect=4/3)
# fig2.fig.savefig(depo_result + '/depo_diurnal.png')
#
# # %%
# fig3, ax = plt.subplots(figsize=(18, 9))
# depo[(depo['value'] < 0.2) & (depo['value'] > -0.05)].groupby('depo_type')['value'].hist(
#     bins=50, ax=ax, alpha=0.5)
# ax.legend(['depo', 'depo_1'])
# ax.set_title('Distribution of depo at max SNR and 1 level below filtered to values in [-0.05, 0.2]')
# ax.set_xlabel('Depo')
# fig3.savefig(depo_result + '/depo_hist.png')
#
# # %%
# fig4, axes = plt.subplots(3, 2, figsize=(18, 9), sharex=True)
# for val, ax in zip(['co_signal', 'co_signal1', 'range',
#                     'vraw', 'beta_raw', 'cross_signal'],
#                    axes.flatten()):
#     ax.plot(depo.groupby('date')[val].mean() if val is not 'beta_raw' else np.log10(
#         depo.groupby('date')[val].mean()), '.')
#     ax.set_title(val)
# fig4.suptitle('Mean values of various metrics at cloud base with max SNR',
#               size=22, weight='bold')
# fig4.savefig(depo_result + '/depo_other_vars.png')
#
# # %%
# fig6, ax = plt.subplots(figsize=(18, 9))
# for name, group in depo.groupby('depo_type'):
#     ax.plot(group.groupby('date').value.mean(), '.', label=name)
# ax.legend()
# ax.set_title('Mean value of depo at max SNR and 1 level below')
# ax.set_xlabel('Date')
# ax.set_ylabel('Depo')
# fig6.savefig(depo_result + '/depo_scatter_ts.png')
#
# # %%
# fig7 = sns.pairplot(depo_original, vars=['range', 'co_signal', 'cross_signal', 'vraw',
#                                          'beta_raw', 'depo'],
#                     height=4.5, aspect=4/3)
# fig7.savefig(depo_result + '/pairplot.png')
#
# # %%
# co_cross_data = depo[['co_signal', 'cross_signal']]
# co_cross_data.dropna(inplace=True)
# H, co_edges, cross_edges = np.histogram2d(co_cross_data['co_signal'] - 1,
#                                           co_cross_data['cross_signal'] - 1,
#                                           bins=500)
# X, Y = np.meshgrid(co_edges, cross_edges)
# fig8, ax = plt.subplots(figsize=(18, 9))
# p = ax.pcolormesh(X, Y, H.T, norm=LogNorm())
# ax.set_xlabel('co_signal - 1')
# ax.set_ylabel('cross_signal - 1')
# colorbar = fig8.colorbar(p, ax=ax)
# colorbar.ax.set_ylabel('Number of observations')
# colorbar.ax.yaxis.set_label_position('left')
# ax.set_title('2D histogram of cross_signal vs co_signal', size=22, weight='bold')
# ax.plot(co_cross_data['co_signal'] - 1,
#         (co_cross_data['co_signal'] - 1) * 0.01, label='depo 0.01 fit',
#         linewidth=0.5)
# ax.plot(co_cross_data['co_signal'] - 1,
#         (co_cross_data['co_signal'] - 1) * 0.07, label='depo 0.07 fit',
#         linewidth=0.5)
# ax.legend(loc='upper left')
# fig8.savefig(depo_result + '/cross_vs_co.png')
