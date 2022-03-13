import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from matplotlib.colors import LogNorm

%matplotlib qt

# %%

file_list = glob.glob(r'F:\halo\paper\figures\background_correction_all/**/*.csv', recursive=True)
df = pd.concat([pd.read_csv(x) for x in file_list[:10]], ignore_index=True)
df['co_corrected'] = df['co_corrected'] - 1

# %%
fig, ax = plt.subplots()
ax.plot(df['co_corrected'], df['depo_wave'] - df['depo'], '.', alpha=0.1)

# %%
df['subtract'] = df['depo_wave'] - df['depo']
x_y_data = df[['co_corrected', 'subtract']].dropna()
H, x_edges, y_edges = np.histogram2d(
    x_y_data['co_corrected'],
    x_y_data['subtract'],
    bins=500)
X, Y = np.meshgrid(x_edges, y_edges)
fig, ax = plt.subplots(figsize=(6, 3))
p = ax.pcolormesh(X, Y, H.T, norm=LogNorm())
ax.set_xlabel('$SNR_{co-corrected}$')
ax.set_ylabel('Depo_corrected - depo_original')
colorbar = fig.colorbar(p, ax=ax)
colorbar.ax.set_ylabel('N')
ax.set_ylim([-0.1, 0.3])
ax.set_xlim([-0.01, 0.2])
# ax.legend(loc='upper left')
