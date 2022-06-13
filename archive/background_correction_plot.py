import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from matplotlib.colors import LogNorm

%matplotlib qt

# %%

file_list = glob.glob(r'F:\halo\paper\figures\background_correction_all/**/*.csv', recursive=True)
df = pd.concat([pd.read_csv(x) for x in file_list], ignore_index=True)
df['co_corrected'] = df['co_corrected'] - 1

# %%
fig, ax = plt.subplots()
ax.plot(df['co_corrected'], df['depo_wave'] - df['depo'], '.', alpha=0.1)

# %%
df['subtract'] = df['depo_wave'] - df['depo']
temp = df[['co_corrected', 'subtract']]
temp = temp[(temp['subtract'] < 0.3) & (temp['co_corrected'] < 0.2)]
temp = temp[(temp['subtract'] > -0.1) & (temp['co_corrected'] > 0)]
x_y_data = temp.dropna()
H, x_edges, y_edges = np.histogram2d(
    x_y_data['co_corrected'],
    x_y_data['subtract'],
    bins=1000)
X, Y = np.meshgrid(x_edges, y_edges)
fig, ax = plt.subplots(figsize=(6, 3))
p = ax.pcolormesh(X, Y, H.T, norm=LogNorm())
ax.set_xlabel('$SNR_{co-corrected}$')
ax.set_ylabel('Depo_corrected - depo_original')
colorbar = fig.colorbar(p, ax=ax)
colorbar.ax.set_ylabel('N')
ax.grid()
# line_x = x_edges[:-1]
# line_y = y_edges[np.argmax(H, axis=1)]
# line_mask = (line_x < 0.1) & (line_x > 0.001)
# ax.plot(line_x[line_mask], line_y[line_mask])
