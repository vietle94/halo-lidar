import numpy as np
import pandas as pd
import os
import glob
import halo_data as hd
from pathlib import Path

# %%
# Define csv directory path
csv_depo_path = r'F:\halo\146\depolarization\depo'
csv_depo_list = [file
                 for path, subdir, files in os.walk(csv_depo_path)
                 for file in glob.glob(os.path.join(path, '*.csv'))]
data_list = glob.glob(r'F:\halo\146\depolarization\*.nc')
noise_list = glob.glob(r'F:\halo\146\depolarization\snr\*_noise.csv')
save_path = r'F:\halo\146\depolarization\depo_saturation'
Path(save_path).mkdir(parents=True, exist_ok=True)

for csv_depo_file in csv_depo_list:
    depo_csv = pd.read_csv(csv_depo_file)
    depo_csv = depo_csv[depo_csv['depo'].notna()]

    depo_date = ''.join(csv_depo_file.split('\\')[-1].split('-')[:3])
    depo_filename = csv_depo_file.split('\\')[-1].replace('_depo.csv', '')
    data_path = [i for i in data_list if depo_date in i][0]
    noise_path = [i for i in noise_list if depo_filename in i][0]

    data = hd.halo_data(data_path)
    # Change masking missing values from -999 to NaN
    data.unmask999()
    # Remove bottom 3 height layers
    data.filter_height()
    noise = pd.read_csv(noise_path, usecols=['noise'])
    noise_threshold = 1 + 3 * np.std(noise['noise'])
    data.filter(variables=['beta_raw', 'v_raw', 'depo_raw'],
                ref='co_signal', threshold=noise_threshold)

    data_time = data.data['time']
    data_range = data.data['range']
    data_cosignal = data.data['co_signal']
    data_depo = data.data['depo_raw']
    data_crosssignal = data.data['cross_signal']
    data_beta = data.data['beta_raw']
    data_vraw = data.data['v_raw']

    for _, row in depo_csv.iterrows():
        mask_time = data_time == row.time
        i_range = np.where(data_range == row.range)[0][0]
        i_range = np.arange(i_range-4, i_range+1)

        co_signal = data_cosignal[mask_time, i_range]
        depo = data_depo[mask_time, i_range]
        cross_signal = data_crosssignal[mask_time, i_range]
        beta = data_beta[mask_time, i_range]
        vraw = data_vraw[mask_time, i_range]
        range = data_range[i_range]
        type = np.arange(-4, 1)

        result = pd.DataFrame.from_dict({
            'year': data.more_info['year'],
            'month': data.more_info['month'],
            'day': data.more_info['day'],
            'location': data.more_info['location'],
            'systemID': data.more_info['systemID'],
            'time': row.time,  # time as hour
            'range': range,  # range
            'depo': depo,  # depo value
            'co_signal': co_signal,  # snr
            'vraw': vraw,  # v_raw
            'beta': beta,  # beta_raw
            'cross_signal': cross_signal,  # cross_signal
            'type': type
        })
        with open(save_path + '/' + row['location'] +
                  '-' + str(int(row['systemID'])) +
                  '_depo_saturation.csv', 'a') as f:
            result.to_csv(f, header=f.tell() == 0, index=False)

# # %%
# i_range
# data_cosignal[mask_time, i_range]
# np.sum(mask_time)
# np.round(row.time, 3)
# data_time
# data_time[787:]
# np.round(data.data['time'][787:], 3)
