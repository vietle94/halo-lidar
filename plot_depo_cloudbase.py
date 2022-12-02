import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import string
from sklearn.mixture import GaussianMixture
import halo_data as hd
from matplotlib.colors import LogNorm
import os
import glob
import matplotlib.dates as dates

# %%
path = r'F:\halo\paper\figures\depo_cloudbase/'


# %%
####################################
# Depo cloudbase hist
####################################
site_names = ['Uto-32', 'Uto-32XR', 'Hyytiala-33',
              'Hyytiala-46', 'Vehmasmaki-53', 'Sodankyla-54']
fig, axes = plt.subplots(3, 2, figsize=(9, 6), sharex=True)
table = pd.read_csv('F:/halo/paper/depo_cloudbase/result.csv')
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
mean = []
std = []
id = []
x_axis = np.linspace(-0.01, 0.1, 100)
for (systemID, group), ax, name in zip(table.groupby('systemID'),
                                       axes.flatten()[1:],
                                       site_names[1:]):
    if systemID == 32:
        x = group['depo'][(group['depo'] > -0.01) & (group['depo'] < 0.1) & (pd.to_datetime(
            group[['year', 'month', 'day']]) < '2017-11-22')]
        axes[0, 0].hist(x, bins=80)

        gmm = GaussianMixture(n_components=2, max_iter=10000)
        gmm.fit(x.values.reshape(-1, 1))
        smean = gmm.means_.ravel()
        sstd = np.sqrt(gmm.covariances_).ravel()
        sort_idx = np.argsort(smean)
        smean = smean[sort_idx]
        sstd = sstd[sort_idx]
        textstr = '\n'.join((
            r'$\mu=%.3f$' % (smean[0], ),
            r'$\sigma=%.3f$' % (sstd[0], )))
        mean.append(smean[0])
        std.append(sstd[0])
        id.append(systemID)
        axes[0, 0].text(0.7, 0.95, textstr, transform=axes[0, 0].transAxes,
                        verticalalignment='top', bbox=props)

        x = group['depo'][(group['depo'] > -0.01) & (group['depo'] < 0.1) & (pd.to_datetime(
            group[['year', 'month', 'day']]) >= '2017-11-22')]
        axes[0, 1].hist(x, bins=80)

        gmm = GaussianMixture(n_components=1, max_iter=10000)
        gmm.fit(x.values.reshape(-1, 1))
        smean = gmm.means_.ravel()
        sstd = np.sqrt(gmm.covariances_).ravel()
        sort_idx = np.argsort(smean)
        smean = smean[sort_idx]
        sstd = sstd[sort_idx]
        textstr = '\n'.join((
            r'$\mu=%.3f$' % (smean[0], ),
            r'$\sigma=%.3f$' % (sstd[0], )))
        mean.append(smean[0])
        std.append(sstd[0])
        id.append(systemID)
        axes[0, 1].text(0.7, 0.95, textstr, transform=axes[0, 1].transAxes,
                        verticalalignment='top', bbox=props)
    elif systemID in [46, 33]:
        x = group['depo'][(group['depo'] > -0.01) & (group['depo'] < 0.1)]
        ax.hist(x, bins=80)

        gmm = GaussianMixture(n_components=2, max_iter=10000)
        gmm.fit(x.values.reshape(-1, 1))
        smean = gmm.means_.ravel()
        sstd = np.sqrt(gmm.covariances_).ravel()
        sort_idx = np.argsort(smean)
        smean = smean[sort_idx]
        sstd = sstd[sort_idx]

        textstr = '\n'.join((
            r'$\mu=%.3f$' % (smean[0], ),
            r'$\sigma=%.3f$' % (sstd[0], )))
        mean.append(smean[0])
        std.append(sstd[0])
        id.append(systemID)
        ax.text(0.7, 0.95, textstr, transform=ax.transAxes,
                verticalalignment='top', bbox=props)
    else:
        x = group['depo'][(group['depo'] > -0.01) & (group['depo'] < 0.1)]
        ax.hist(x, bins=80)
        gmm = GaussianMixture(n_components=1, max_iter=10000)
        gmm.fit(x.values.reshape(-1, 1))
        smean = gmm.means_.ravel()
        sstd = np.sqrt(gmm.covariances_).ravel()
        sort_idx = np.argsort(smean)
        smean = smean[sort_idx]
        sstd = sstd[sort_idx]
        textstr = '\n'.join((
            r'$\mu=%.3f$' % (smean[0], ),
            r'$\sigma=%.3f$' % (sstd[0], )))
        mean.append(smean[0])
        std.append(sstd[0])
        id.append(systemID)
        ax.text(0.7, 0.95, textstr, transform=ax.transAxes,
                verticalalignment='top', bbox=props)
for n, ax in enumerate(axes.flatten()):
    ax.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
            transform=ax.transAxes, size=12)
    ax.set_ylabel('N')
    ax.grid()
axes.flatten()[-2].set_xlabel('$\delta$')
axes.flatten()[-1].set_xlabel('$\delta$')
fig.subplots_adjust(hspace=0.5, wspace=0.3)
fig.savefig(path + '/depo_cloudbase.png', dpi=150,
            bbox_inches='tight')

# %%
####################################
# Depo cloudbase ts
####################################
site_names = ['Uto-32', 'Uto-32XR', 'Hyytiala-33',
              'Hyytiala-46', 'Vehmasmaki-53', 'Sodankyla-54']
fig, axes = plt.subplots(3, 2, figsize=(9, 8), sharey=True)
table = pd.read_csv('F:/halo/paper/depo_cloudbase/result.csv')
minute, table['hour'] = np.modf(table['time'])
second, table['minute'] = np.modf(minute * 60)
table['second'] = second * 60
table['time_'] = pd.to_datetime(table[['year', 'month', 'day', 'hour', 'minute', 'second']])

group_ = table.groupby('systemID')
for id, ax, name in zip([32, 33, 46, 53, 54],
                        axes.flatten()[1:],
                        site_names[1:]):
    if id == 32:
        group = group_.get_group(id)
        x = group[(group['depo'] > -0.01) & (group['depo'] < 0.1) & (pd.to_datetime(
            group[['year', 'month', 'day']]) < '2017-11-22')]
        x_group = x.groupby(x['time_'].dt.date)['depo']
        x_group_mean = x_group.mean()
        axes[0, 0].errorbar(x_group_mean.index, x_group.mean(),
                            yerr=x_group.std(), ls='none', marker='.', linewidth=0.5, markersize=5)

        x = group[(group['depo'] > -0.01) & (group['depo'] < 0.1) & (pd.to_datetime(
            group[['year', 'month', 'day']]) >= '2017-11-22')]
        x_group = x.groupby(x['time_'].dt.date)['depo']
        x_group_mean = x_group.mean()
        axes[0, 1].errorbar(x_group_mean.index, x_group.mean(),
                            yerr=x_group.std(), ls='none', marker='.', linewidth=0.5, markersize=5)
    else:
        group = group_.get_group(id)
        x = group[(group['depo'] > -0.01) & (group['depo'] < 0.1)]
        x_group = x.groupby(x['time_'].dt.date)['depo']
        x_group_mean = x_group.mean()
        ax.errorbar(x_group_mean.index, x_group.mean(),
                    yerr=x_group.std(), ls='none', marker='.', linewidth=0.5, markersize=5)

for n, ax in enumerate(axes.flatten()):
    ax.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
            transform=ax.transAxes, size=12)
    ax.tick_params(axis='x', labelrotation=45)
    ax.grid()
fig.subplots_adjust(hspace=0.7)


# %%
####################################
# Depo cloudbase combined
####################################
table = pd.read_csv('F:/halo/paper/depo_cloudbase/result.csv')
minute, table['hour'] = np.modf(table['time'])
second, table['minute'] = np.modf(minute * 60)
table['second'] = second * 60
table['time_'] = pd.to_datetime(table[['year', 'month', 'day', 'hour', 'minute', 'second']])
group_ = table.groupby('systemID')


# %%
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
fig = plt.figure(figsize=(12, 12))
gs = fig.add_gridspec(nrows=6, ncols=10, wspace=0.7, hspace=0.5)
ax0 = fig.add_subplot(gs[0, :8])
ax0_hist = fig.add_subplot(gs[0, 8:], sharey=ax0)

ax1 = fig.add_subplot(gs[1, :8], sharey=ax0)
ax1_hist = fig.add_subplot(gs[1, 8:], sharey=ax0)

ax2 = fig.add_subplot(gs[2, :8], sharey=ax0)
ax2_hist = fig.add_subplot(gs[2, 8:], sharey=ax0)

ax3 = fig.add_subplot(gs[3, :8], sharey=ax0)
ax3_hist = fig.add_subplot(gs[3, 8:], sharey=ax0)

ax4 = fig.add_subplot(gs[4, :8], sharey=ax0)
ax4_hist = fig.add_subplot(gs[4, 8:], sharey=ax0)

ax5 = fig.add_subplot(gs[5, :8], sharey=ax0)
ax5_hist = fig.add_subplot(gs[5, 8:], sharey=ax0)

for ax in [ax0, ax1, ax2, ax3, ax5]:
    ax.sharex(ax4)

for id, ax, ax_hist in zip([32, 33, 46, 53, 54],
                           [ax1, ax2, ax3, ax4, ax5],
                           [ax1_hist, ax2_hist, ax3_hist, ax4_hist, ax5_hist]):
    if id == 32:
        group = group_.get_group(id)
        x = group[(group['depo'] > -0.01) & (group['depo'] < 0.1) & (pd.to_datetime(
            group[['year', 'month', 'day']]) < '2017-11-22')]
        print(id, x['depo'].mean(), x['depo'].std())
        x_group = x.groupby(x['time_'].dt.date)['depo']
        x_group_mean = x_group.mean()
        ax0.errorbar(x_group_mean.index, x_group.mean(),
                     yerr=x_group.std(), ls='none', marker='.', linewidth=0.5, markersize=5)
        ax0_hist.hist(x['depo'], bins=50, orientation='horizontal')
        gmm = GaussianMixture(n_components=2, max_iter=10000)
        gmm.fit(x['depo'].values.reshape(-1, 1))
        smean = gmm.means_.ravel()
        sstd = np.sqrt(gmm.covariances_).ravel()
        sort_idx = np.argsort(smean)
        smean = smean[sort_idx]
        sstd = sstd[sort_idx]
        textstr = '\n'.join((
            r'$\mu=%.3f$' % (smean[0], ),
            r'$\sigma=%.3f$' % (sstd[0], )))
        ax0_hist.text(0.5, 0.95, textstr, transform=ax0_hist.transAxes,
                      verticalalignment='top', bbox=props)

        x = group[(group['depo'] > -0.01) & (group['depo'] < 0.1) & (pd.to_datetime(
            group[['year', 'month', 'day']]) >= '2017-11-22')]
        print(id, x['depo'].mean(), x['depo'].std())
        x_group = x.groupby(x['time_'].dt.date)['depo']
        x_group_mean = x_group.mean()
        ax1.errorbar(x_group_mean.index, x_group.mean(),
                     yerr=x_group.std(), ls='none', marker='.', linewidth=0.5, markersize=5)
        ax1_hist.hist(x['depo'], bins=50, orientation='horizontal')
        gmm = GaussianMixture(n_components=1, max_iter=10000)
        gmm.fit(x['depo'].values.reshape(-1, 1))
        smean = gmm.means_.ravel()
        sstd = np.sqrt(gmm.covariances_).ravel()
        sort_idx = np.argsort(smean)
        smean = smean[sort_idx]
        sstd = sstd[sort_idx]
        textstr = '\n'.join((
            r'$\mu=%.3f$' % (smean[0], ),
            r'$\sigma=%.3f$' % (sstd[0], )))
        ax1_hist.text(0.5, 0.95, textstr, transform=ax1_hist.transAxes,
                      verticalalignment='top', bbox=props)
    elif id in [46, 33]:
        group = group_.get_group(id)
        x = group[(group['depo'] > -0.01) & (group['depo'] < 0.1)]
        print(id, x['depo'].mean(), x['depo'].std())
        x_group = x.groupby(x['time_'].dt.date)['depo']
        x_group_mean = x_group.mean()
        ax.errorbar(x_group_mean.index, x_group.mean(),
                    yerr=x_group.std(), ls='none', marker='.', linewidth=0.5, markersize=5)
        ax_hist.hist(x['depo'], bins=50, orientation='horizontal')
        gmm = GaussianMixture(n_components=2, max_iter=10000)
        gmm.fit(x['depo'].values.reshape(-1, 1))
        smean = gmm.means_.ravel()
        sstd = np.sqrt(gmm.covariances_).ravel()
        sort_idx = np.argsort(smean)
        smean = smean[sort_idx]
        sstd = sstd[sort_idx]
        textstr = '\n'.join((
            r'$\mu=%.3f$' % (smean[0], ),
            r'$\sigma=%.3f$' % (sstd[0], )))
        ax_hist.text(0.5, 0.95, textstr, transform=ax_hist.transAxes,
                     verticalalignment='top', bbox=props)
    else:
        group = group_.get_group(id)
        x = group[(group['depo'] > -0.01) & (group['depo'] < 0.1)]
        print(id, x['depo'].mean(), x['depo'].std())
        x_group = x.groupby(x['time_'].dt.date)['depo']
        x_group_mean = x_group.mean()
        ax.errorbar(x_group_mean.index, x_group.mean(),
                    yerr=x_group.std(), ls='none', marker='.', linewidth=0.5, markersize=5)
        ax_hist.hist(x['depo'], bins=50, orientation='horizontal')
        gmm = GaussianMixture(n_components=1, max_iter=10000)
        gmm.fit(x['depo'].values.reshape(-1, 1))
        smean = gmm.means_.ravel()
        sstd = np.sqrt(gmm.covariances_).ravel()
        sort_idx = np.argsort(smean)
        smean = smean[sort_idx]
        sstd = sstd[sort_idx]
        textstr = '\n'.join((
            r'$\mu=%.3f$' % (smean[0], ),
            r'$\sigma=%.3f$' % (sstd[0], )))
        ax_hist.text(0.5, 0.95, textstr, transform=ax_hist.transAxes,
                     verticalalignment='top', bbox=props)

for n, ax in enumerate([ax0, ax1, ax2, ax3, ax4, ax5]):
    ax.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
            transform=ax.transAxes, size=12)
    ax.tick_params(axis='x', labelrotation=0)
    ax.set_yticks([0, 0.025, 0.05])
    ax.grid()
    ax.xaxis.set_major_locator(dates.MonthLocator([6, 12]))
    ax.set_ylabel('$\delta$')

for ax in [ax0_hist, ax1_hist, ax2_hist, ax3_hist, ax4_hist, ax5_hist]:
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.set_xlabel('N')
    ax.grid()
ax0.set_ylim([-0.01, 0.07])
# fig.subplots_adjust(hspace=0.7)
# fig.savefig(path + '/depo_ts.png', dpi=500,
#             bbox_inches='tight')

# %%
######################################
# Depo profile
######################################

i_plot = 206807
site = str(int(table.iloc[i_plot]['systemID']))
if site == '32':
    data = hd.getdata('F:/halo/' + site + '/depolarization/normal') + \
        hd.getdata('F:/halo/' + site + '/depolarization/xr')
else:
    data = hd.getdata('F:/halo/' + site + '/depolarization/')
date = ''.join(list(map(lambda x: str(int(table.iloc[i_plot][x])).zfill(2),
                        ['year', 'month', 'day'])))
file = [file for file in data if date in file][0]

df = hd.halo_data(file)
df.filter_height()
df.unmask999()
df.filter(variables=['beta_raw', 'depo_raw'],
          ref='co_signal',
          threshold=1 + 3 * df.snr_sd)

# mask_range_plot = (df.data['range'] <= table.iloc[i_plot]['range'] + 100) & (
#     df.data['range'] >= table.iloc[i_plot]['range'] - 100)
cloud_base_height = table.iloc[i_plot]['range']
mask_range_plot = (df.data['range'] <= cloud_base_height + 150)
mask_time_plot = df.data['time'] == table.iloc[i_plot]['time']
depo_profile_plot = df.data['depo_raw'][mask_time_plot,
                                        mask_range_plot]
co_signal_profile_plot = df.data['co_signal'][mask_time_plot,
                                              mask_range_plot]
beta_profile_plot = df.data['beta_raw'][mask_time_plot,
                                        mask_range_plot]

fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=True)

for ax, var, lab in zip(axes.flatten(), ['depo_raw', 'co_signal', 'beta_raw'],
                        ['$\delta$', '$SNR_{co}$', r'$\beta\quad[Mm^{-1}]$']):
    for h, leg in zip([df.data['range'] <= cloud_base_height,
                       df.data['range'] == cloud_base_height,
                       (cloud_base_height < df.data['range']) &
                       (df.data['range'] <= cloud_base_height + 150)],
                      ['Aerosol', 'Cloud base', 'In-cloud']):
        ax.plot(df.data[var][mask_time_plot, h],
                df.data['range'][h], '.', label=leg)
    ax.grid()
    # ax.axhline(y=cloud_base_height, linestyle='--', color='grey')
    ax.set_xlabel(lab)
axes[2].set_xscale('log')
axes[0].set_xlim([-0.05, 0.1])
# axes[2].set_xlim([1e-8, 1e-3])
axes[0].set_ylabel('Height a.g.l [m]')

for n, ax in enumerate(axes.flatten()):
    ax.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
            transform=ax.transAxes, size=12)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3)
fig.subplots_adjust(bottom=0.2)
print(df.filename)
fig.savefig(path + df.filename + '_depo_profile.png',
            bbox_inches='tight', dpi=150)

# %%
######################################
# Co, cross saturation
######################################
# Define csv directory path
depo_paths = [
    'F:\\halo\\32\\depolarization\\depo\\xr']

# Collect csv file in csv directory and subdirectory
result_list = []
for csv_path in depo_paths:
    data_list = [file
                 for path, subdir, files in os.walk(csv_path)
                 for file in glob.glob(os.path.join(path, '*.csv'))]
    result_list.extend(data_list)

# %%
depo = pd.concat([pd.read_csv(f) for f in result_list],
                 ignore_index=True)
depo = depo.astype({'year': int, 'month': int, 'day': int, 'systemID': int})
depo = depo.astype({'systemID': str})
# For right now, just take the date, ignore hh:mm:ss
depo['date'] = pd.to_datetime(depo[['year', 'month', 'day']])
depo.drop(['depo_1', 'co_signal1'], axis=1, inplace=True)
depo.loc[(depo['date'] > '2017-11-20') & (depo['systemID'] == '32'),
         'systemID'] = '32XR'

depo['sys'] = depo['location'] + '-' + depo['systemID']

# %%
fig, ax = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
table32 = table[(table['systemID'] == 32) & (pd.to_datetime(
    table[['year', 'month', 'day']]) >= '2017-11-22')]

co_cross_data = table32[['co_signal', 'cross_signal']].dropna()
H, co_edges, cross_edges = np.histogram2d(
    co_cross_data['co_signal'] - 1,
    co_cross_data['cross_signal'] - 1,
    bins=500)


y_hat = (co_cross_data['co_signal'] - 1) * 0.004
text = "$SNR_{cross}=0.004\;SNR_{co}$"
ax[0].plot(co_cross_data['co_signal'] - 1,
           y_hat, "r-", lw=1, label=text)

# ax[0].text(0.05, 0.95, text, transform=ax[0].transAxes,
#         fontsize=10, verticalalignment='top', bbox=props)
ax[0].legend(fontsize=10, facecolor='wheat', framealpha=0.5)

X, Y = np.meshgrid(co_edges, cross_edges)
p = ax[0].pcolormesh(X, Y, H.T, norm=LogNorm())
ax[0].set_xlabel('$SNR_{co}$')
ax[0].set_ylabel('$SNR_{cross}$')
colorbar = fig.colorbar(p, ax=ax[0])
colorbar.ax.set_ylabel('N')

for key, group in depo.groupby('sys'):
    # Co-cross histogram
    if key == 'Uto-32XR':
        co_cross_data = group[['co_signal', 'cross_signal']].dropna()
        H, co_edges, cross_edges = np.histogram2d(
            co_cross_data['co_signal'] - 1,
            co_cross_data['cross_signal'] - 1,
            bins=1000)
        y_hat = (co_cross_data['co_signal'] - 1) * 0.004
        text = "$SNR_{cross}=0.004\;SNR_{co}$"
        ax[1].plot(co_cross_data['co_signal'] - 1,
                   y_hat, "r-", lw=1, label=text)
        ax[1].legend(fontsize=10, facecolor='wheat', framealpha=0.5)

        # ax[1].text(0.05, 0.95, text, transform=ax[1].transAxes,
        #         fontsize=10, verticalalignment='top', bbox=props)
        X, Y = np.meshgrid(co_edges, cross_edges)
        p = ax[1].pcolormesh(X, Y, H.T, norm=LogNorm())
        ax[1].set_xlabel('$SNR_{co}$')
        colorbar = fig.colorbar(p, ax=ax[1])
        colorbar.ax.set_ylabel('N')
        ax[1].set_ylim(top=0.3)
        break
for n, ax_ in enumerate(ax.flatten()):
    ax_.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
             transform=ax_.transAxes, size=12)
    ax_.grid()
fig.savefig(path + 'co_cross_saturation.png', dpi=150,
            bbox_inches='tight')

# %%
