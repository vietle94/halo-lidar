import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import halo_data as hd
from pathlib import Path

# %%
threshold = -5
time_save = []
range_save = []
depo_save = []
year = []
month = []
day = []
location = []
systemID = []
width = -3
for site in ['32', '33', '46', '53', '54']:
    if site == '32':
        data = hd.getdata('F:/halo/' + site + '/depolarization/normal') + \
            hd.getdata('F:/halo/' + site + '/depolarization/xr')
    else:
        data = hd.getdata('F:/halo/' + site + '/depolarization/')

    for csv_filename in Path('F:/halo/' + site + '/depolarization/depo').rglob('*.csv'):
        df_ = pd.read_csv(csv_filename)
        nc_filename = ''.join(str(csv_filename).split('\\')[-1].split('-')[:3])

        file = [file for file in data if nc_filename in file][0]
        print(file)

        df = hd.halo_data(file)
        df.filter_height()
        df.unmask999()
        df.filter(variables=['beta_raw', 'depo_raw'],
                  ref='co_signal',
                  threshold=1 + 3 * df.snr_sd)

        for i in range(len(df_)):
            mask_time = df.data['time'] == df_['time'][i]
            mask_range = df.data['range'] <= df_['range'][i]
            depo_profile = df.data['depo_raw'][mask_time, mask_range]
            depo_below_cloud = depo_profile[width:]
            range_below_cloud = df.data['range'][mask_range][width:]

            # check for monotonically increasing
            difference = np.diff(depo_below_cloud)
            if ((difference[~np.isnan(difference)]) < 0).any():
                continue

            beta_below_cloud = df.data['beta_raw'][mask_time, mask_range][width:]
            m_ = np.log10(beta_below_cloud) > threshold
            depo_below_cloud2 = depo_below_cloud[(m_) & (~np.isnan(depo_below_cloud))]
            range_below_cloud = range_below_cloud[(m_) & (~np.isnan(depo_below_cloud))]

            if len(depo_below_cloud2) > 0:
                i_min = np.argmin(depo_below_cloud2)
                time_save.append(df_['time'][i])
                range_save.append(range_below_cloud[i_min])
                depo_save.append(depo_below_cloud2[i_min])
                year.append(df_['year'][0])
                month.append(df_['month'][0])
                day.append(df_['day'][0])
                location.append(df_['location'][0])
                systemID.append(df_['systemID'][0])

                # mask_range_plot = (df.data['range'] <= range_below_cloud[i_min] + 100) & (
                #     df.data['range'] >= range_below_cloud[i_min] - 200)
                # depo_profile_plot = df.data['depo_raw'][df.data['time'] == df_['time'][i],
                #                                         mask_range_plot]
                # co_signal_profile_plot = df.data['co_signal'][df.data['time'] == df_['time'][i],
                #                                               mask_range_plot]

                # fig = plt.figure(figsize=(18, 9))
                # ax1 = fig.add_subplot(211)
                # ax2 = fig.add_subplot(223)
                # ax3 = fig.add_subplot(224, sharey=ax2)
                # c = ax1.pcolormesh(df.data['time'], df.data['range'],
                #                    np.log10(df.data['beta_raw']).T, cmap='jet', vmin=-8, vmax=-4)
                # cbar = fig.colorbar(c, ax=ax1, fraction=0.01)
                # cbar.ax.set_ylabel('Beta', rotation=90)
                # cbar.ax.yaxis.set_label_position('left')
                # ax1.set_title(df.filename, weight='bold', size=22)
                # ax1.set_xlabel('Time (h)')
                # ax1.set_xlim([0, 24])
                # ax1.set_ylim([0, None])
                # ax1.set_ylabel('Height (km)')
                # ax1.yaxis.set_major_formatter(hd.m_km_ticks())
                # ax1.axvline(df_['time'][i], color='red')
                # ax1.text(df_['time'][i], -.05, 'Data', color='red',
                #          transform=ax1.get_xaxis_transform(),
                #          ha='center', va='top')
                #
                # ax2.plot(depo_profile_plot, df.data['range'][mask_range_plot], '.')
                # ax2.axhline(y=range_below_cloud[i_min], linestyle='--')
                # ax2.set_xlim([-0.1, 0.5])
                # ax2.set_xlabel('Depolarization ratio')
                # ax2.set_ylabel('Height (km)')
                # ax3.plot(co_signal_profile_plot - 1,
                #          df.data['range'][mask_range_plot], '.')
                # ax3.axhline(y=range_below_cloud[i_min], linestyle='--')
                # ax3.set_xlabel('CO_SNR')
                # ax3.set_ylabel('Height (km)')
                # fig.savefig('F:/halo/' + site + '/depolarization/depo/result/' +
                #             df.filename + '_' + str(df_['time'][i]) + '_cloudbase.png',
                #             bbox_inches='tight')
                # plt.close('all')

result = pd.DataFrame.from_dict({
    'year': year,
    'month': month,
    'day': day,
    'location': location,
    'systemID': systemID,
    'time': time_save,  # time as hour
    'range': range_save,  # range
    'depo': depo_save,  # depo value
})

with open('F:/halo/paper/depo_cloudbase/result.csv', 'w') as f:
    result.to_csv(f, header=f.tell() == 0, index=False)

# # %%
# fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
# df = pd.read_csv('F:/halo/' + '32' + '/depolarization/depo/result/result.csv')
#
# axes[0, 0].hist(df['depo'][(df['depo'] > -0.02) & (df['depo'] < 0.03) & (pd.to_datetime(
#     df[['year', 'month', 'day']]) < '2018-01-01')], bins=20)
# axes[0, 0].set_title('32', weight='bold')
#
# axes[0, 1].hist(df['depo'][(df['depo'] > -0.02) & (df['depo'] < 0.03) & (pd.to_datetime(
#     df[['year', 'month', 'day']]) >= '2018-01-01')], bins=20)
# axes[0, 1].set_title('32XR', weight='bold')
#
# for ax, site in zip(axes.flatten()[2:], ['33', '46', '53', '54']):
#     df = pd.read_csv('F:/halo/' + site + '/depolarization/depo/result/result.csv')
#     ax.hist(df['depo'][(df['depo'] > -0.02) & (df['depo'] < 0.03)], bins=20)
#     ax.set_title(site, weight='bold')
# fig.subplots_adjust(hspace=0.5)
#
# # %%
# fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
# df = pd.read_csv('F:/halo/paper/depo_cloudbase/result.csv')
# for (systemID, group), ax in zip(df.groupby('systemID'), axes.flatten()[1:]):
#     if systemID == 32:
#         axes[0, 0].hist(group['depo'][(group['depo'] > -0.01) & (group['depo'] < 0.1) & (pd.to_datetime(
#             group[['year', 'month', 'day']]) < '2017-11-22')], bins=20)
#         axes[0, 0].set_title('32', weight='bold')
#
#         axes[0, 1].hist(group['depo'][(group['depo'] > -0.01) & (group['depo'] < 0.1) & (pd.to_datetime(
#             group[['year', 'month', 'day']]) >= '2017-11-22')], bins=20)
#         axes[0, 1].set_title('32XR', weight='bold')
#     else:
#         ax.hist(group['depo'][(group['depo'] > -0.01) & (group['depo'] < 0.1)], bins=20)
#         ax.set_title(int(systemID), weight='bold')
# fig.subplots_adjust(hspace=0.5)
# fig.savefig('5width_filtered2.png', bbox_inches='tight')

# # %%
# fig, ax = plt.subplots(figsize=(12, 9), sharex=True)
# df = pd.read_csv('F:/halo/summary/result.csv')
# group = df[df['systemID'] == 46]
# ax.hist(group['depo'][(group['depo'] > -0.01) & (group['depo'] < 0.15)], bins=20)
# ax.set_title(int(systemID), weight='bold')
# fig.subplots_adjust(hspace=0.5)
