import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import halo_data as hd
from pathlib import Path
import pandas as pd
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# %%
# Specify data folder path
data_folder = r'F:\halo\32\depolarization'
# Specify folder path for snr collection
snr_folder = data_folder + r'\snr'
# Specify output images folder path
image_folder = data_folder + r'\aerosol_algorithm'
Path(image_folder).mkdir(parents=True, exist_ok=True)

# %%
file_list = glob.glob(data_folder + '/*.nc')
noise_list = glob.glob(snr_folder + '/*_noise.csv')

# %%
date_range = pd.date_range(start='2016-01-01', end='2016-02-01').strftime('%Y%m%d')

# %%

depo_raw = np.array([])
v_raw = np.array([])
beta_raw = np.array([])
date_raw = np.array([])
range_raw = {}
time_raw = {}

for date in date_range:
    file = [file for file in file_list if date in file]
    if len(file) == 0:
        print(f'{date} is missing')
        continue
    elif len(file) > 1:
        print(f'There are two {date}')
        break

    file = file[0]

    df = hd.halo_data(file)
    df.unmask999()
    df.filter_height()

    noise = pd.read_csv([noise_file for noise_file in noise_list
                         if df.filename in noise_file][0],
                        usecols=['noise'])
    noise_threshold = 1 + 3 * np.std(noise['noise'])
    df.filter(variables=['beta_raw', 'v_raw', 'depo_raw'],
              ref='co_signal', threshold=noise_threshold)
    df.filter_attenuation(variables=['beta_raw', 'v_raw', 'depo_raw'],
                          ref='beta_raw', threshold=10**-4.5, buffer=2)
    depo_raw = np.concatenate([depo_raw,
                               df.data['depo_raw'][:, :100].flatten()])
    v_raw = np.concatenate([v_raw,
                            df.data['v_raw'][:, :100].flatten()])

    b = df.data['beta_raw'][:, :100].flatten()
    beta_raw = np.concatenate([beta_raw,
                               b])
    date_raw = np.concatenate([date_raw,
                               np.repeat(date, len(b))])
    date_raw = date_raw.astype('int')
    range_raw[date] = df.data['range'][:100]
    time_raw[date] = df.data['time']

X_full = np.vstack([depo_raw, v_raw, beta_raw, date_raw])
X_full = X_full.T
X_full[X_full[:, 0] < 0, 0] = np.nan
X = X_full[~np.isnan(X_full).any(axis=1)]

# %%
X[X[:, 0] > 1, 0] = 1
X[:, 2] = np.log10(X[:, 2])
X[X[:, 1] > 2, 1] = 2
X[X[:, 1] < -2, 1] = -2

# %%
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[:, :3])
kmeans = KMeans(n_clusters=5, max_iter=3000).fit(X_scaled)

# %%
label = np.repeat(np.nan, X_full.shape[0])
label[~np.isnan(X_full).any(axis=1)] = kmeans.labels_

# %%
scaler.inverse_transform(kmeans.cluster_centers_)

# %%
date = '20160102'
fig, ax = plt.subplots(4, 1, figsize=(12, 9))
p = ax[0].pcolormesh(time_raw[date], range_raw[date],
                     np.log10(X_full[X_full[:, 3] == int(date), 2].reshape(int(len(time_raw[date])),
                                                                           int(len(range_raw[date]))).T),
                     cmap='jet', vmin=-8, vmax=-4)
fig.colorbar(p, ax=ax[0])
p = ax[1].pcolormesh(time_raw[date], range_raw[date],
                     X_full[X_full[:, 3] == int(date), 1].reshape(int(len(time_raw[date])),
                                                                  int(len(range_raw[date]))).T,
                     cmap='jet', vmin=-2, vmax=2)
fig.colorbar(p, ax=ax[1])
p = ax[2].pcolormesh(time_raw[date], range_raw[date],
                     X_full[X_full[:, 3] == int(date), 0].reshape(int(len(time_raw[date])),
                                                                  int(len(range_raw[date]))).T,
                     cmap='jet', vmin=0, vmax=0.5)
fig.colorbar(p, ax=ax[2])
p = ax[3].pcolormesh(time_raw[date], range_raw[date],
                     label[X_full[:, 3] == int(date)].reshape(int(len(time_raw[date])),
                                                              int(len(range_raw[date]))).T,
                     cmap=plt.cm.get_cmap('jet', 5))
fig.colorbar(p, ax=ax[3])
fig.tight_layout()
