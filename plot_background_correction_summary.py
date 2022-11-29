import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
import string

# %%
sites = ['33', '46', '53', '54']
file_paths = {x: glob.glob(
    r'F:\halo\paper\figures\background_correction_all/stan/' + x + '/' + '**/*.csv') for x in sites}
site_32 = glob.glob(r'F:\halo\paper\figures\background_correction_all/stan/' +
                    '32' + '/' + '**/*.csv')
site_32 = sorted(site_32)
file_paths['32'] = site_32[:513]
file_paths['32XR'] = site_32[513:]

site_plot = ['32', '32XR', '33', '46', '53', '54']

# %%
fig, axes = plt.subplots(3, 2, figsize=(9, 6), sharey=True, sharex=True)
for key, ax in zip(site_plot, axes.flatten()):
    value = file_paths[key]
    df = pd.concat([pd.read_csv(x) for x in value], ignore_index=True)
    df['time'] = pd.to_datetime(df['time'])

    df['subtract'] = df['depo_corrected'] - df['depo']
    temp = df[['co_corrected', 'subtract']]
    temp['co_corrected'] = temp['co_corrected'] - 1
    temp = temp[(temp['subtract'] < 0.3) & (temp['co_corrected'] < 0.2)]
    temp = temp[(temp['subtract'] > -0.1) & (temp['co_corrected'] > 0)]
    x_y_data = temp.dropna()
    H, x_edges, y_edges = np.histogram2d(
        x_y_data['co_corrected'],
        x_y_data['subtract'], bins=400)
    X, Y = np.meshgrid(x_edges, y_edges)
    H[H < 1] = np.nan
    p = ax.pcolormesh(X, Y, H.T)
    ax.set_xlim([0, 0.03])
    ax.set_xticks([0, 0.01, 0.02, 0.03])
    colorbar = fig.colorbar(p, ax=ax)
    colorbar.ax.set_ylabel('N')
    # line_x = x_edges[:-1]
    # line_y = y_edges[np.argmax(H, axis=1)]
    # line_mask = (line_x < 0.1) & (line_x > 0.001)
    # ax.plot(line_x[line_mask], line_y[line_mask], c='red')
    # temp = df[['co_corrected', 'depo_corrected_sd']]
    # temp['co_corrected'] = temp['co_corrected'] - 1
    # temp = temp[(temp['depo_corrected_sd'] < 0.1) & (temp['co_corrected'] < 0.2)]
    # temp = temp[(temp['depo_corrected_sd'] > -0.1) & (temp['co_corrected'] > 0)]
    # x_y_data = temp.dropna()
    # H, x_edges, y_edges = np.histogram2d(
    #     x_y_data['depo_corrected_sd'],
    #     x_y_data['co_corrected'],
    #     bins=500)
    # X, Y = np.meshgrid(x_edges, y_edges)
    # H[H < 1] = np.nan
    # p = ax[1].pcolormesh(X, Y, H.T)
    # ax[0].set_ylabel('$SNR_{co,corrected}$')
    # ax[1].set_ylim([0, 0.03])
    # colorbar = fig.colorbar(p, ax=ax[1])
    # colorbar.ax.set_ylabel('N')
    # # line_x = x_edges[:-1]
    # # line_y = y_edges[np.nanargmax(H, axis=1)]
    # # line_mask = (line_x < 0.1) & (line_x > 0.001)
    # # ax[1].plot(line_x[line_mask], line_y[line_mask], c='red')

axes[0, 0].set_ylabel(r'$\delta_{corrected} - \delta_{original}$')
axes[1, 0].set_ylabel(r'$\delta_{corrected} - \delta_{original}$')
axes[2, 0].set_ylabel(r'$\delta_{corrected} - \delta_{original}$')
axes[-1, 0].set_xlabel('$SNR_{co, corrected}$')
axes[-1, 1].set_xlabel('$SNR_{co, corrected}$')
ax.set_ylim([-0.1, 0.3])
for n, ax_ in enumerate(axes.flatten()):
    ax_.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
             transform=ax_.transAxes, size=12)
    ax_.grid()

fig.savefig(r'F:\halo\paper\figures\background_correction_all/summary.png',
            dpi=150, bbox_inches='tight')

# %%
for key, ax in zip(site_plot, axes.flatten()):
    value = file_paths[key]
    df = pd.concat([pd.read_csv(x) for x in value], ignore_index=True)
    df['time'] = pd.to_datetime(df['time'])

    df['subtract'] = df['depo_corrected'] - df['depo']
    temp = df[['co_corrected', 'subtract']]
    temp['co_corrected'] = temp['co_corrected'] - 1
    temp = temp[(temp['subtract'] < 0.3) & (temp['co_corrected'] < 0.2)]
    temp = temp[(temp['subtract'] > -0.1) & (temp['co_corrected'] > 0)]
    temp = temp.dropna()
    mynum = temp[np.abs(temp['subtract']) > 0.05].size/temp.size
    mynum2 = temp[np.abs(temp['subtract']) > 0.01].size/temp.size
    mynum3 = temp[np.abs(temp['subtract']) > 0.1].size/temp.size
    print(key, mynum, mynum2, mynum3)
