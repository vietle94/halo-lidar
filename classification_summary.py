import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np
import seaborn as sns
from scipy.stats import binned_statistic_2d
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import calendar
import os
%matplotlib qt

# %%
classifier_folder = 'F:\\halo\\classifier\\32'
files = glob.glob(classifier_folder + '/*.csv')

# subyear = {'1st': [1, 2, 3, 4],
#            '2nd': [5, 6, 7, 8],
#            '3rd': [9, 10, 11, 12]}
#
#
# def file_subyear(subyear, sub, year, files):
#     list_files = []
#     patterns = [str(year) + '-' + str(s).zfill(2) for s in subyear[sub]]
#     for file in files:
#         for pattern in patterns:
#             if pattern in file:
#                 list_files.append(file)
#     return list_files
#
#
# # %%
# year = 2018
# sub = "3rd"
# list_files = file_subyear(subyear, sub, year, files)

# %%
bin_height = np.arange(0, 10000, 300)
bin_time = np.arange(0, 24.5, 0.5)

for file in files:
    df = pd.read_csv(file,
                     usecols=['date', 'location',
                              'co_signal', 'cross_signal',
                              'time', 'range', 'classifier'])
    aerosol = df[df['classifier'] // 10 == 1]

    if (aerosol.shape[0] == 0) or (aerosol.shape[0] == 1):
        continue

    co, _, _, _ = binned_statistic_2d(aerosol['range'],
                                      aerosol['time'],
                                      aerosol['co_signal'],
                                      bins=[bin_height, bin_time],
                                      statistic=np.nanmean)
    cross, _, _, _ = binned_statistic_2d(aerosol['range'],
                                         aerosol['time'],
                                         aerosol['cross_signal'],
                                         bins=[bin_height, bin_time],
                                         statistic=np.nanmean)
    depo = (cross-1)/(co-1)

    count_aerosol, _, _, _ = binned_statistic_2d(aerosol['range'],
                                                 aerosol['time'],
                                                 aerosol['co_signal'],
                                                 bins=[bin_height, bin_time],
                                                 statistic='count')
    count, _, _, _ = binned_statistic_2d(df['range'],
                                         df['time'],
                                         df['co_signal'],
                                         bins=[bin_height, bin_time],
                                         statistic='count')

    depo[count_aerosol/count < 0.5] = np.nan

    depo_save = depo.flatten()
    time_save = bin_time[:-1]
    range_save = np.repeat(bin_height[:-1], depo.shape[1])

    dff = pd.DataFrame({'depo': depo.flatten(), 'time': np.tile(time_save, depo.shape[0]),
                        'range': range_save,
                        'date': np.repeat(df['date'][0], depo_save.size),
                        'location': np.repeat(df['location'][0], depo_save.size)})
    dff.dropna(inplace=True)

    save_file = classifier_folder + '/result.csv'
    dff.to_csv(save_file, mode='a', header=not os.path.exists(save_file), index=False)

# %%
aerosol

# %%
fig, ax = plt.subplots()
ax.pcolormesh(bin_time[:-1], bin_height[:-1], depo, cmap='jet', vmax=0.5, vmin=0)

# %%
fig, ax = plt.subplots()
p = ax.pcolormesh(bin_time[:-1], bin_height[:-1], count_aerosol/count, cmap='jet')
fig.colorbar(p, ax=ax)

# %%
classifier = pd.concat(
    [pd.read_csv(f,
                 usecols=['date', 'location',
                          'co_signal', 'cross_signal',
                          'time', 'range', 'classifier']) for f in list_files],
    ignore_index=True)

classifier = classifier.replace([np.inf, -np.inf], np.nan)
classifier['date'] = pd.to_datetime(classifier['date'])

temp = classifier
temp = temp[temp['classifier'] // 10 == 1]
bin_height = np.arange(0, temp['range'].max() + 300, 300)
bin_time = np.arange(0, 24+0.5, 0.5)

df = pd.DataFrame({'date': [], 'depo': []})
for key, grp in temp.groupby('date'):
    if grp.shape[0] == 1:
        depo = (grp['cross_signal']-1)/(grp['co_signal']-1)
        depo = depo[depo < 0.8]
        depo = depo[depo > -0.25]
        dff = pd.DataFrame({'date': key, 'depo': depo})
        df = df.append(dff)
    else:
        co, _, _, _ = binned_statistic_2d(grp['range'],
                                          grp['time'],
                                          grp['co_signal'],
                                          bins=[bin_height, bin_time],
                                          statistic=np.nanmean)
        cross, _, _, _ = binned_statistic_2d(grp['range'],
                                             grp['time'],
                                             grp['cross_signal'],
                                             bins=[bin_height, bin_time],
                                             statistic=np.nanmean)
        depo = (cross-1)/(co-1)
        depo = depo[depo < 0.8]
        depo = depo[depo > -0.25]
        dff = pd.DataFrame({'date': key, 'depo': depo})
        df = df.append(dff)

cmap = cm.Greens
norm = Normalize(vmin=20000, vmax=200000)
n0 = temp.groupby('date')['cross_signal'].count()

n0 = n0.reset_index()
n0['month'] = n0.date.dt.month
n0['day'] = n0.date.dt.day
n1 = {}
for k, v in n0.groupby('month'):
    newdict = v[['day', 'cross_signal']].set_index('day')['cross_signal'].to_dict()
    for key, value in newdict.items():
        newdict[key] = cmap(norm(value))
    n1[k] = newdict

newdf = df.copy()
newdf['day'] = newdf['date'].dt.day
newdf['month'] = newdf['date'].dt.month
# fig, axes = plt.subplots(4, 1, figsize=(11, 15))
# with sns.axes_style("darkgrid"):
#     for mo, ax in zip(subyear[sub], axes.flatten()):
#         if not (newdf['month'] == mo).any():
#             continue
#         sns.boxplot('day', 'depo', data=newdf[newdf['month'] == mo], ax=ax,
#                     palette=n1[mo], showfliers=False, linewidth=1)
#         # ax.plot(newdf[newdf['month'] == mo].groupby('day')['depo'].median().values,
#         #         color='#ff7f0e')
#         ax.set_ylim([-0.2, 0.6])
#         ax.set_xlabel('')
#         ax.set_ylabel('Depolarization ratio')
#         ax.set_title(calendar.month_name[mo], size=14, weight='bold')
#     sns.despine(left=True)
# fig.subplots_adjust(hspace=0.4)
# # fig.savefig('F:\\halo\\classifier\\result\\' + sub + str(year) +
#             classifier['location'][0] + '.png', bbox_inches='tight')

# %%
newdf.reset_index().to_csv(sub+'.csv')

# %%
df = pd.concat(
    [pd.read_csv(i,
                 usecols=['date', 'depo',
                          'day', 'month']) for
     i in ['1st.csv', '2nd.csv', '3rd.csv']], ignore_index=True)


df
# %%
cmap = cm.Greens
norm = Normalize(vmin=1035, vmax=10898)
n0 = df.groupby('month')['depo'].count()

n0 = n0.reset_index()
# n0['month'] = n0.date.dt.month
# n0['day'] = n0.date.dt.day
n1 = {}
for k, v in n0.groupby('month'):
    newdict = v[['day', 'cross_signal']].set_index('day')['cross_signal'].to_dict()
    for key, value in newdict.items():
        newdict[key] = cmap(norm(value))
    n1[k] = newdict


# %%
n0 = n0.set_index('month')['depo'].to_dict()
n0
n1 = {key: cmap(norm(val)) for key, val in n0.items()}
n1

# %%
fig, ax = plt.subplots(figsize=(12, 6))
sns.violinplot('month', 'depo', data=df, ax=ax,
               palette=n1, inner='box', cut=-5, bw=0.2, linewidth=1)
ax.set_title('Time series of depo in Hyytiala-2018', size=22, weight='bold')
# ax.set_ylim([-0.2, 0.5])
fig.savefig('ts.png', dpi=200, bbox_inches='tight')
