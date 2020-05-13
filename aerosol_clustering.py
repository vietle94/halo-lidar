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
for month in np.arange(1, 13):
    month_pattern = '2016' + str(month).zfill(2)
    print(month_pattern)
    days_month = [file for file in file_list if month_pattern in file]

    depo_raw = np.array([])
    v_raw = np.array([])
    beta_raw = np.array([])

    for file in days_month:
        df = hd.halo_data(file)

        df.unmask999()
        df.filter_height()
        noise = pd.read_csv(noise_list[df.filename in noise_list],
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
        beta_raw = np.concatenate([beta_raw,
                                   df.data['beta_raw'][:, :100].flatten()])
    X = np.vstack([depo_raw, v_raw, beta_raw])
    X = X.T
    X = X[~np.isnan(X).any(axis=1)]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=5, max_iter=3000).fit(X_scaled)

    for file in days_month:
        data = hd.halo_data(file)
        data.unmask999()
        data.filter_height()
        noise = pd.read_csv(noise_list[df.filename in noise_list],
                            usecols=['noise'])
        noise_threshold = 1 + 3 * np.std(noise['noise'])
        data.filter(variables=['beta_raw', 'v_raw', 'depo_raw'],
                    ref='co_signal', threshold=noise_threshold)
        fig, axes = plt.subplots(4, 1, figsize=(18, 9))
        for ax, var, name in zip(axes[:3],
                                 [np.log10(data.data['beta_raw'][:, :100].T),
                                  data.data['v_raw'][:, :100].T,
                                  data.data['depo_raw'][:, :100].T],
                                 ['beta_raw', 'v_raw', 'depo_raw']):
            pp = ax.pcolormesh(data.data['time'].data,
                               data.data['range'][:100],
                               var,
                               cmap='jet',
                               vmin=data.cbar_lim[name][0],
                               vmax=data.cbar_lim[name][1])
            fig.colorbar(pp, ax=ax)
            ax.set_title(name)

        data.filter_attenuation(variables=['beta_raw', 'v_raw', 'depo_raw'],
                                ref='beta_raw', threshold=10**-4.5, buffer=2)

        depo_raw = data.data['depo_raw'][:, :100].flatten()
        v_raw = data.data['v_raw'][:, :100].flatten()
        beta_raw = data.data['beta_raw'][:, :100].flatten()

        X_data = np.vstack([depo_raw, v_raw, beta_raw])
        X_data = X_data.T
        X_data_scaled = scaler.transform(X_data)
        label = np.full([X_data.shape[0], ], np.nan)
        for i, val in enumerate(X_data_scaled):
            if not np.isnan(val).any():
                label[i] = kmeans.predict(val.reshape(1, -1))

        shape1 = data.data['time'].shape[0]
        shape2 = 100

        label_reshaped = label.reshape([shape1, shape2])
        beta_reshaped = beta_raw.reshape([shape1, shape2])

        p = axes[3].pcolormesh(data.data['time'].data,
                               data.data['range'][:100],
                               label_reshaped.T, vmin=0, vmax=5,
                               cmap=plt.cm.get_cmap('jet', 5))
        axes[3].set_title('resulted clutering')
        fig.colorbar(p, ax=axes[3])
        fig.suptitle(data.filename, size=22, weight='bold')
        fig.savefig(image_folder + '/' + data.filename + '.png')

    full_label = kmeans.predict(X)
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.hist(full_label)
    ax.set_title(f'Number of observations per cluster in month {month}')
    ax.set_xlabel('Cluster label')
    ax.set_yscale('log')
    fig.savefig(image_folder + '/' + str(month) + '.png')

    # Save centroid coordinate
    centroid = scaler.inverse_transform(kmeans.cluster_centers_)
    centroid_lab = kmeans.predict(kmeans.cluster_centers_)
    centroid_df = pd.DataFrame(centroid, columns=['depo', 'v_raw', 'beta'])
    centroid_df['label'] = centroid_lab
    centroid_df['month'] = month
    centroid_df.to_csv(image_folder + '/' + str(month) + '.csv', index=False)
