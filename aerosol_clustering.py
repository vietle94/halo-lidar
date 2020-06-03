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
from scipy.stats import binned_statistic
from sklearn.mixture import GaussianMixture

# %%
# Specify data folder path
data_folder = r'F:\halo\32\depolarization'
# Specify folder path for snr collection
snr_folder = data_folder + r'\snr'
# Specify output images folder path
# image_folder = data_folder + r'\aerosol_algorithm'
# Path(image_folder).mkdir(parents=True, exist_ok=True)

# %%
X_full, time_full, range_full = hd.aggregate_data(data_folder, snr_folder,
                                                  '2016-01-01', '2016-01-31',
                                                  height_lim=100, snr_mul=3,
                                                  cloud_thres=10**-4.5,
                                                  cloud_buffer=2,
                                                  attenuation=True,
                                                  positive_depo=True)

# %%
na_mask = ~np.isnan(X_full).any(axis=1)
X = X_full[na_mask]

# %%
X.loc[X['depo'] > 1, 'depo'] = 1
X.loc[:, 'beta_raw'] = np.log10(X.loc[:, 'beta_raw'])

X.loc[X['v_raw'] > 2, 'v_raw'] = 2
X.loc[X['v_raw'] < -2, 'v_raw'] = -2

# %%
gmm = GaussianMixture(n_components=5, max_iter=1000)
gmm.fit(X.iloc[:, :3])

# %%
prob = gmm.predict_proba(X.iloc[:, :3])
prob_lab = gmm.predict(X.iloc[:, :3])
delta_prob = (prob.max(axis=1)[:, np.newaxis] - prob)
for i, row in enumerate(delta_prob):
    row_ = row[row != 0]
    if min(row_) < 0.1:
        prob_lab[i] = 5

# %%
label = np.repeat(np.nan, X_full.shape[0])
label[na_mask] = prob_lab
gmm.means_
gmm.covariances_


gmm.predict(gmm.means_)

# %%
date = '20160115'
beta_date = X_full.loc[X_full['date'] == int(date), 'beta_raw'].values
depo_date = X_full.loc[X_full['date'] == int(date), 'depo'].values
v_date = X_full.loc[X_full['date'] == int(date), 'v_raw'].values

time_date = time_full[date]
time_dim = int(len(time_date))
range_date = range_full[date]
range_dim = int(len(range_date))
fig, ax = plt.subplots(4, 1, figsize=(12, 9))
p = ax[0].pcolormesh(time_date, range_date,
                     np.log10(beta_date.reshape(time_dim, range_dim).T),
                     cmap='jet', vmin=-8, vmax=-4)
fig.colorbar(p, ax=ax[0])
p = ax[1].pcolormesh(time_date, range_date,
                     v_date.reshape(time_dim, range_dim).T,
                     cmap='jet', vmin=-2, vmax=2)
fig.colorbar(p, ax=ax[1])
p = ax[2].pcolormesh(time_date, range_date,
                     depo_date.reshape(time_dim, range_dim).T,
                     cmap='jet', vmin=0, vmax=0.5)
fig.colorbar(p, ax=ax[2])
p = ax[3].pcolormesh(time_date, range_date,
                     label[X_full['date'] == int(date)].reshape(time_dim, range_dim).T,
                     cmap=plt.cm.get_cmap('jet', 6))
cbar = fig.colorbar(p, ax=ax[3])
cbar.set_ticks([0, 1, 2, 3, 4, 5])
cbar.set_ticklabels(['Heavy aerosol', 'Light aerosol', 'Snow fall',
                     'Rain fall', 'Liquid cloud/Drizzle', 'None-defined'])
fig.tight_layout()

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
date = '20160115'
beta_date = X_full.loc[X_full['date'] == int(date), 'beta_raw'].values
depo_date = X_full.loc[X_full['date'] == int(date), 'depo'].values
v_date = X_full.loc[X_full['date'] == int(date), 'v_raw'].values

time_date = time_full[date]
time_dim = int(len(time_date))
range_date = range_full[date]
range_dim = int(len(range_date))
fig, ax = plt.subplots(4, 1, figsize=(12, 9))
p = ax[0].pcolormesh(time_date, range_date,
                     np.log10(beta_date.reshape(time_dim, range_dim).T),
                     cmap='jet', vmin=-8, vmax=-4)
fig.colorbar(p, ax=ax[0])
p = ax[1].pcolormesh(time_date, range_date,
                     v_date.reshape(time_dim, range_dim).T,
                     cmap='jet', vmin=-2, vmax=2)
fig.colorbar(p, ax=ax[1])
p = ax[2].pcolormesh(time_date, range_date,
                     depo_date.reshape(time_dim, range_dim).T,
                     cmap='jet', vmin=0, vmax=0.5)
fig.colorbar(p, ax=ax[2])
p = ax[3].pcolormesh(time_date, range_date,
                     label[X_full['date'] == int(date)].reshape(time_dim, range_dim).T,
                     cmap=plt.cm.get_cmap('jet', 6))
cbar = fig.colorbar(p, ax=ax[3])
cbar.set_ticks([0, 1, 2, 3, 4, 5])
cbar.set_ticklabels(['Heavy aerosol', 'Light aerosol', 'Snow fall',
                     'Rain fall', 'Liquid cloud/Drizzle', 'None-defined'])
fig.tight_layout()

# %%
df = hd.halo_data(r'F:\halo\32\depolarization/20160117_fmi_halo-doppler-lidar-32-depolarization.nc')
df.unmask999()
df.filter_height()
noise = pd.read_csv(r'F:\halo\32\depolarization\snr/2016-01-17-Uto-32_noise.csv',
                    usecols=['noise'])
noise_threshold = 1 + 3 * np.std(noise['noise'])
df.filter(variables=['beta_raw', 'v_raw', 'depo_raw'],
          ref='co_signal', threshold=noise_threshold)
b = df.data['beta_raw'][:, :100]
b = np.log10(b)
v = df.data['v_raw'][:, :100]
d = df.data['depo_raw'][:, :100]

# %%
d[d > 1] = 1

d[d < 0] = np.nan
v[d < 0] = np.nan
b[d < 0] = np.nan

v[v > 2] = 2
v[v < -2] = -2

# %%


def nan_ptp(a):
    return np.ptp(a[np.isfinite(a)])


b_255 = (255*(b - np.nanmin(b))/nan_ptp(b))
v_255 = (255*(v - np.nanmin(v))/nan_ptp(v))
d_255 = (255*(d - np.nanmin(d))/nan_ptp(d))

b_255[np.isnan(b_255)] = 255
v_255[np.isnan(v_255)] = 255
d_255[np.isnan(d_255)] = 255


im = np.zeros((100, 952, 3))
im[:, :, 0] = b_255.astype(int).T
im[:, :, 1] = v_255.astype(int).T
im[:, :, 2] = d_255.astype(int).T

im = im.astype(int)
# %%
fig, ax = plt.subplots(figsize=(12, 9))
ax.imshow(im, origin='lower', aspect=1.9)
