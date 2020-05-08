import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import halo_data as hd
from pathlib import Path
import pandas as pd

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
# for month in np.arange(2, 13):
month = 1
month_pattern = '2016' + str(month).zfill(2)
days_month = [file for file in file_list if month_pattern in file]

depo_raw = np.array([])
v_raw = np.array([])
beta_raw = np.array([])
co_signal = np.array([])
cross_signal = np.array([])

for file in days_month:
    df = hd.halo_data(file)

    df.unmask999()
    df.filter_height()
    noise = pd.read_csv(noise_list[df.filename in noise_list],
                        usecols=['noise'])
    noise_threshold = 1 + 3 * np.std(noise['noise'])
    df.filter(variables=['beta_raw', 'v_raw', 'cross_signal', 'depo_raw'],
              ref='co_signal', threshold=noise_threshold)
    depo_raw = np.concatenate([depo_raw, df.data['depo_raw'][:, :100].flatten()])
    v_raw = np.concatenate([v_raw, df.data['v_raw'][:, :100].flatten()])
    beta_raw = np.concatenate([beta_raw, df.data['beta_raw'][:, :100].flatten()])
    co_signal = np.concatenate([co_signal, df.data['co_signal'][:, :100].flatten()])
    cross_signal = np.concatenate([cross_signal, df.data['cross_signal'][:, :100].flatten()])

# %%
X = np.vstack([depo_raw, v_raw, beta_raw, co_signal, cross_signal])
X = X.T
X = X[~np.isnan(X).any(axis=1)]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=5, max_iter=3000).fit(X_scaled)
for file in days_month:
    data = hd.halo_data(file)
    data.unmask999()
    data.filter_height()
    data.filter(variables=['beta_raw', 'v_raw', 'cross_signal', 'depo_raw'],
                ref='co_signal', threshold=1+0.0005*3)
    depo_raw = data.data['depo_raw'][:, :100].flatten()
    v_raw = data.data['v_raw'][:, :100].flatten()
    beta_raw = data.data['beta_raw'][:, :100].flatten()
    co_signal = data.data['co_signal'][:, :100].flatten()
    cross_signal = data.data['cross_signal'][:, :100].flatten()

    X_data = np.vstack([depo_raw, v_raw, beta_raw, co_signal, cross_signal])
    X_data = X_data.T
    scaler = StandardScaler()
    X_data_scaled = scaler.fit_transform(X_data)
    label = np.full([X_data.shape[0], ], np.nan)
    for i, val in enumerate(X_data_scaled):
        if not np.isnan(val).any():
            label[i] = kmeans.predict(val.reshape(1, -1))

    shape1 = data.data['time'].shape[0]
    shape2 = 100

    label_reshaped = label.reshape([shape1, shape2])
    beta_reshaped = beta_raw.reshape([shape1, shape2])
    count = np.unique(label, return_counts=True)
    data.data['beta_raw'][:, :100].shape

    fig, ax = plt.subplots(2, 1, figsize=(18, 9))
    pp = ax[0].pcolormesh(data.data['time'].data,
                          data.data['range'][:100],
                          np.log10(data.data['beta_raw'][:, :100].T),
                          cmap='jet', vmin=-8, vmax=-4)
    fig.colorbar(pp, ax=ax[0])
    ax[0].set_title('beta_raw')
    p = ax[1].pcolormesh(data.data['time'].data,
                         data.data['range'][:100],
                         label_reshaped.T, vmin=0, vmax=5,
                         cmap=plt.cm.get_cmap('jet', 5))
    ax[1].set_title('resulted clutering')
    fig.colorbar(p, ax=ax[1])
    fig.suptitle(data.filename, size=22, weight='bold')
    # fig.savefig(image_folder + '/' + data.filename + '.png')
    break

full_label = kmeans.predict(X)

fig, ax = plt.subplots(figsize=(16, 9))
ax.hist(full_label)
ax.set_title(f'Number of observations per cluster in {month} month')
ax.set_xlabel('Cluster label')
# fig.savefig(image_folder + '/' + str(month) + '.png')
