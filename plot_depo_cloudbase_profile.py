import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import string
import halo_data as hd

# %%
path = r'F:\halo\paper\figures\depo_cloudbase/'


# %%
####################################
# Depo cloudbase hist
####################################
site_names = ['Uto-32', 'Uto-32XR', 'Hyytiala-33',
              'Hyytiala-46', 'Vehmasmaki-53', 'Sodankyla-54']
table = pd.read_csv('F:/halo/paper/depo_cloudbase/result.csv')
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

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
cross_signal_profile_plot = df.data['cross_signal'][mask_time_plot,
                                                    mask_range_plot]
beta_profile_plot = df.data['beta_raw'][mask_time_plot,
                                        mask_range_plot]
depo_sd_profile_plot = depo_profile_plot * \
    np.sqrt(
        ((df.snr_sd/cross_signal_profile_plot-1))**2 +
        ((df.snr_sd/co_signal_profile_plot-1))**2)

# %%
mask_time_avg = (df.data['time'] > np.floor(table.iloc[i_plot]['time'])) & (
    df.data['time'] < (np.floor(table.iloc[i_plot]['time'] + 1)))
mask_range_plot_avg = (df.data['range'] < cloud_base_height)
co_signal_profile_avg = df.data['co_signal'][mask_time_avg].mean(axis=0)[mask_range_plot_avg]
co_count = df.data['co_signal'][mask_time_avg].shape[0]
cross_signal_profile_avg = np.nanmean(df.data['cross_signal'][mask_time_avg], axis=0)[
    mask_range_plot_avg]
cross_count = df.data['cross_signal'][mask_time_avg].shape[0]

depo_profile_avg = (cross_signal_profile_avg - 1) / (co_signal_profile_avg - 1)
depo_sd_profile_avg = np.abs(depo_profile_avg) * \
    np.sqrt(
        ((df.snr_sd/np.sqrt(cross_count))/(cross_signal_profile_avg-1))**2 +
        ((df.snr_sd/np.sqrt(co_count))/(co_signal_profile_avg-1))**2)

# %%
fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=True)

for ax, var, lab in zip(axes.flatten(), ['depo_raw', 'co_signal', 'beta_raw'],
                        ['$\delta$', '$SNR_{co}$', r"$\beta'\quad[m^{-1}sr^{-1}]$"]):
    for h, leg in zip([df.data['range'] <= cloud_base_height,
                       df.data['range'] == cloud_base_height,
                       (cloud_base_height < df.data['range']) &
                       (df.data['range'] <= cloud_base_height + 150)],
                      ['Aerosol', 'Cloud base', 'In-cloud']):
        if lab == r'$\delta$':
            depo_profile_plot = df.data[var][mask_time_plot, h]
            co_signal_profile_plot = df.data['co_signal'][mask_time_plot, h]
            cross_signal_profile_plot = df.data['cross_signal'][mask_time_plot, h]
            depo_sd_profile_plot = np.abs(depo_profile_plot) * \
                np.sqrt(
                    (df.snr_sd/(cross_signal_profile_plot-1))**2 +
                    (df.snr_sd/(co_signal_profile_plot-1))**2)
            if leg == 'Aerosol':
                ax.errorbar(depo_profile_plot,
                            df.data['range'][h],
                            xerr=depo_sd_profile_plot,
                            fmt='.',
                            errorevery=1, elinewidth=0.7,
                            alpha=0.3, ms=6,
                            label=leg)
                ax.errorbar(depo_profile_avg,
                            df.data['range'][mask_range_plot_avg],
                            xerr=depo_sd_profile_avg,
                            fmt='.',
                            errorevery=1, elinewidth=1,
                            alpha=1, ms=6, c='tab:blue',
                            label='1h averaged aerosol')
            else:
                ax.errorbar(depo_profile_plot,
                            df.data['range'][h],
                            xerr=depo_sd_profile_plot,
                            fmt='.',
                            errorevery=1, elinewidth=0.5,
                            alpha=0.7, ms=6,
                            label=leg)
            # ax.plot(depo_profile_plot, df.data['range'][h], '.', label=leg)

        elif lab == r'$SNR_{co}$':
            ax.plot(df.data[var][mask_time_plot, h] - 1,
                    df.data['range'][h], '.', label=leg)
        else:
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
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4)
fig.subplots_adjust(bottom=0.2)
print(df.filename)
fig.savefig(path + df.filename + '_depo_profile.png',
            bbox_inches='tight', dpi=600)
