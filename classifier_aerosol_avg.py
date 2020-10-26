import numpy as np
import pandas as pd
import glob
from scipy.stats import binned_statistic_2d

# %%
for site in ['46', '54', '33', '53', '34', '32']:
    path = 'F:\\halo\\classifier2\\' + site + '\\'
    list_files = glob.glob(path + '*.csv')

    for file in list_files:
        df = pd.read_csv(file)
        date = df['date'][0]
        location = df['location'][0]
        df['co_signal'] = df['co_signal'].replace({-999: np.nan})
        df = df[~np.isnan(df['co_signal'])]

        bin_time = np.arange(0, 24+0.5, 0.5)
        bin_range = np.arange(0, np.nanmax(df['range'])+300, 300)

        count_all, _, _, _ = binned_statistic_2d(
            df['time'],
            df['range'],
            df['classifier'],
            bins=[bin_time, bin_range],
            statistic='count')

        count_aerosol, _, _, _ = binned_statistic_2d(
            df['time'],
            df['range'],
            (df['classifier'] // 10 == 1),
            bins=[bin_time, bin_range],
            statistic=np.sum)

        co_avg, _, _, _ = binned_statistic_2d(
            df['time'],
            df['range'],
            df['co_signal'],
            bins=[bin_time, bin_range],
            statistic=np.nanmean)

        cross_avg, _, _, _ = binned_statistic_2d(
            df['time'],
            df['range'],
            df['cross_signal'],
            bins=[bin_time, bin_range],
            statistic=np.nanmean)

        depo_avg = cross_avg/co_avg
        depo_avg[count_aerosol/count_all < 0.5] = np.nan
        range_save, time_save = np.meshgrid(bin_range[:-1]+150,
                                            bin_time[:-1]+0.25)
        df_save = pd.DataFrame({
            'depo': depo_avg.flatten(),
            'time': time_save.flatten(),
            'range': range_save.flatten(),
            'date': date,
            'location': location
        })

        df_save.dropna(inplace=True)

        with open(path + '/' 'result.csv', 'a') as f:
            df_save.to_csv(f, header=f.tell() == 0, index=False)
