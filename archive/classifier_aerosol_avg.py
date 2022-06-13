import numpy as np
import pandas as pd
import glob
from scipy.stats import binned_statistic_2d

# %%
bin_time = np.arange(0, 24+0.5, 0.5)
for site in ['46', '54', '33', '53', '34', '32']:
    path = 'F:\\halo\\classifier2\\' + site + '\\'
    list_files = glob.glob(path + '*.csv')
    df_save = pd.DataFrame()
    for file in list_files:
        df = pd.read_csv(file)
        date = df['date'][0]
        location = df['location'][0]
        df['co_signal'] = df['co_signal'].replace({-999: np.nan})
        df = df[~np.isnan(df['co_signal'])]

        bin_range = np.arange(0, np.nanmax(df['range'])+300, 300)
        df_aerosol = df[(df['classifier'] // 10 == 1)]
        if len(df_aerosol.index) == 0:
            continue
        thres_aerosol, _, _, _ = binned_statistic_2d(
            df['time'],
            df['range'],
            (df['classifier'] // 10 == 1),
            bins=[bin_time, bin_range],
            statistic=np.nanmean)

        co_avg, _, _, _ = binned_statistic_2d(
            df_aerosol['time'],
            df_aerosol['range'],
            df_aerosol['co_signal'],
            bins=[bin_time, bin_range],
            statistic=np.nanmean)

        cross_avg, _, _, _ = binned_statistic_2d(
            df_aerosol['time'],
            df_aerosol['range'],
            df_aerosol['cross_signal'],
            bins=[bin_time, bin_range],
            statistic=np.nanmean)

        depo_avg = (cross_avg-1)/(co_avg-1)
        depo_avg[thres_aerosol < 0.5] = np.nan
        range_save, time_save = np.meshgrid(bin_range[:-1]+150,
                                            bin_time[:-1]+0.25)
        df_save0 = pd.DataFrame({
            'depo': depo_avg.flatten(),
            'time': time_save.flatten(),
            'range': range_save.flatten(),
            'date': date,
            'location': location
        })

        df_save0.dropna(inplace=True)
        df_save = df_save.append(df_save0, ignore_index=True)
        print(date + location)
    df_save.to_csv(path + '/result.csv', index=False)
