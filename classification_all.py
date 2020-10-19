from scipy.ndimage import median_filter
from scipy.ndimage import maximum_filter
from sklearn.cluster import DBSCAN
import numpy as np
import halo_data as hd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl
from scipy.stats import binned_statistic_2d
import pandas as pd


data = hd.getdata('F:/halo/46/depolarization')
classifier_folder = 'F:\\halo\\classifier2\\46'
Path(classifier_folder).mkdir(parents=True, exist_ok=True)

# date = '2017'
files = [file for file in data]
for file in files:
    df = hd.halo_data(file)
    df.filter_height()
    df.unmask999()
    df.depo_cross_adj()

    df.filter(variables=['beta_raw'],
              ref='co_signal',
              threshold=1 + df.snr_sd)

    # Save before further filtering
    beta_save = df.data['beta_raw'].flatten()
    depo_save = df.data['depo_adj'].flatten()
    co_save = df.data['co_signal'].flatten()
    cross_save = df.data['cross_signal'].flatten()  # Already adjusted with bleed

    df.data['classifier'] = np.zeros(df.data['beta_raw'].shape, dtype=int)

    log_beta = np.log10(df.data['beta_raw'])
    # Aerosol
    aerosol = log_beta < -5.5

    # Small size median filter to remove noise
    aerosol_smoothed = median_filter(aerosol, size=11)
    # Remove thin bridges, better for the clustering
    aerosol_smoothed = median_filter(aerosol_smoothed, size=(15, 1))

    df.data['classifier'][aerosol_smoothed] = 10

    df.filter(variables=['beta_raw', 'v_raw', 'depo_adj'],
              ref='co_signal',
              threshold=1 + 3 * df.snr_sd)
    log_beta = np.log10(df.data['beta_raw'])

    range_save = np.tile(df.data['range'],
                         df.data['beta_raw'].shape[0])

    time_save = np.repeat(df.data['time'],
                          df.data['beta_raw'].shape[1])
    v_save = df.data['v_raw'].flatten()  # put here to avoid noisy values at 1sd snr

    # Liquid
    liquid = log_beta > -5.5

    # maximum filter to increase the size of liquid region
    liquid_max = maximum_filter(liquid, size=5)
    # Median filter to remove background noise
    liquid_smoothed = median_filter(liquid_max, size=13)

    df.data['classifier'][liquid_smoothed] = 30

    # updraft - indication of aerosol zone
    updraft = df.data['v_raw'] > 1
    updraft_smooth = median_filter(updraft, size=3)
    updraft_max = maximum_filter(updraft_smooth, size=91)

    # Fill the gap in aerosol zone
    updraft_median = median_filter(updraft_max, size=31)

    # precipitation < -1 (center of precipitation)
    precipitation_1 = (log_beta > -7) & (df.data['v_raw'] < -1)

    precipitation_1_median = median_filter(precipitation_1, size=9)

    # Only select precipitation outside of aerosol zone
    precipitation_1_ne = precipitation_1_median * ~updraft_median
    precipitation_1_median_smooth = median_filter(precipitation_1_ne,
                                                  size=3)
    precipitation = precipitation_1_median_smooth

    # precipitation < -0.5 (include all precipitation)
    precipitation_1_low = (log_beta > -7) & (df.data['v_raw'] < -0.5)

    # Avoid ebola infection surrounding updraft
    # Useful to contain error during ebola precipitation
    updraft_ebola = df.data['v_raw'] > 0.2
    updraft_ebola_max = maximum_filter(updraft_ebola, size=3)

    # Ebola precipitation
    for _ in range(1500):
        prep_1_max = maximum_filter(precipitation, size=3)
        prep_1_max *= ~updraft_ebola_max  # Avoid updraft area
        precipitation_ = precipitation_1_low * prep_1_max
        if np.sum(precipitation) == np.sum(precipitation_):
            break
        precipitation = precipitation_

    df.data['classifier'][precipitation] = 20

    # Remove all aerosol above cloud or precipitation
    mask_aerosol0 = df.data['classifier'] == 10
    for i in np.array([20, 30]):
        if i == 20:
            mask = df.data['classifier'] == i
        else:
            mask = log_beta > -5
            mask = maximum_filter(mask, size=5)
            mask = median_filter(mask, size=13)
        mask_row = np.argwhere(mask.any(axis=1)).reshape(-1)
        mask_col = np.nanargmax(mask[mask_row, :], axis=1)
        for row, col in zip(mask_row, mask_col):
            mask[row, col:] = True
        mask_undefined = mask * mask_aerosol0
        df.data['classifier'][mask_undefined] = i

    # %%
    if (df.data['classifier'] == 10).any():
        classifier = df.data['classifier'].ravel()
        time_dbscan = np.repeat(np.arange(df.data['time'].size),
                                df.data['beta_raw'].shape[1])
        height_dbscan = np.tile(np.arange(df.data['range'].size),
                                df.data['beta_raw'].shape[0])

        time_dbscan = time_dbscan[classifier == 10].reshape(-1, 1)
        height_dbscan = height_dbscan[classifier == 10].reshape(-1, 1)
        X = np.hstack([time_dbscan, height_dbscan])
        db = DBSCAN(eps=3, min_samples=25, n_jobs=-1).fit(X)

        v_dbscan = v_save[classifier == 10]
        range_dbscan = range_save[classifier == 10]

        v_dict = {}
        r_dict = {}
        for i in np.unique(db.labels_):
            v_dict[i] = np.nanmean(v_dbscan[db.labels_ == i])
            r_dict[i] = np.nanmin(range_dbscan[db.labels_ == i])

        lab = db.labels_.copy()
        for key, val in v_dict.items():
            if key == -1:
                lab[db.labels_ == key] = 40
            elif (val < -0.5):
                lab[db.labels_ == key] = 20
            elif r_dict[key] == min(df.data['range']):
                lab[db.labels_ == key] = 10
            elif (val > -0.2):
                lab[db.labels_ == key] = 11
            else:
                lab[db.labels_ == key] = 40

        df.data['classifier'][df.data['classifier'] == 10] = lab

    # %%
    # Separate ground rain
    if (df.data['classifier'] == 20).any():
        classifier = df.data['classifier'].ravel()
        time_dbscan = np.repeat(np.arange(df.data['time'].size),
                                df.data['beta_raw'].shape[1])
        height_dbscan = np.tile(np.arange(df.data['range'].size),
                                df.data['beta_raw'].shape[0])

        time_dbscan = time_dbscan[classifier == 20].reshape(-1, 1)
        height_dbscan = height_dbscan[classifier == 20].reshape(-1, 1)
        X = np.hstack([time_dbscan, height_dbscan])
        db = DBSCAN(eps=3, min_samples=1, n_jobs=-1).fit(X)

        range_dbscan = range_save[classifier == 20]

        r_dict = {}
        for i in np.unique(db.labels_):
            r_dict[i] = np.nanmin(range_dbscan[db.labels_ == i])

        lab = db.labels_.copy()
        for key, val in r_dict.items():
            if r_dict[key] == min(df.data['range']):
                lab[db.labels_ == key] = 20
            else:
                lab[db.labels_ == key] = 30

        df.data['classifier'][df.data['classifier'] == 20] = lab

    # %%
    cmap = mpl.colors.ListedColormap(
        ['white', '#2ca02c', '#808000', 'blue', 'red', 'gray'])
    boundaries = [0, 10, 11, 20, 30, 40, 50]
    norm = mpl.colors.BoundaryNorm(boundaries, cmap.N, clip=True)

    # %%
    classifier = df.data['classifier'].flatten()
    result = pd.DataFrame({'date': df.date,
                           'location': df.location,
                           'beta': np.log10(beta_save),
                           'v': v_save,
                           'depo': depo_save,
                           'co_signal': co_save,
                           'cross_signal': cross_save,
                           'time': time_save,
                           'range': range_save,
                           'classifier': classifier})

    result.to_csv(classifier_folder + '/' + df.filename + '_classified.csv',
                  index=False)

    # %%
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(4, 2, width_ratios=[5, 4])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0], sharex=ax1, sharey=ax1)
    ax4 = fig.add_subplot(gs[1, 1], sharex=ax2)
    ax5 = fig.add_subplot(gs[2, 0], sharex=ax1, sharey=ax1)
    ax6 = fig.add_subplot(gs[2, 1], sharex=ax2)
    ax7 = fig.add_subplot(gs[3, 0], sharex=ax1, sharey=ax1)
    ax8 = fig.add_subplot(gs[3, 1], sharex=ax2)
    p1 = ax1.pcolormesh(df.data['time'], df.data['range'],
                        np.log10(df.data['beta_raw']).T, cmap='jet', vmin=-8, vmax=-4)
    p2 = ax3.pcolormesh(df.data['time'], df.data['range'],
                        df.data['v_raw'].T, cmap='jet', vmin=-2, vmax=2)
    p3 = ax5.pcolormesh(df.data['time'], df.data['range'],
                        df.data['depo_adj'].T, cmap='jet', vmin=0, vmax=0.5)
    p4 = ax7.pcolormesh(df.data['time'], df.data['range'],
                        df.data['classifier'].T,
                        cmap=cmap, norm=norm)
    for ax in [ax1, ax3, ax5, ax7]:
        ax.yaxis.set_major_formatter(hd.m_km_ticks())
        ax.set_ylabel('Height [km, a.g.l]')

    cbar = fig.colorbar(p1, ax=ax1)
    cbar.ax.set_ylabel('Attenuated backscatter')
    cbar.ax.yaxis.set_label_position('left')
    cbar = fig.colorbar(p2, ax=ax3)
    cbar.ax.set_ylabel('Velocity [m/s]')
    cbar.ax.yaxis.set_label_position('left')
    cbar = fig.colorbar(p3, ax=ax5)
    cbar.ax.set_ylabel('Depolarization ratio')
    cbar.ax.yaxis.set_label_position('left')
    cbar = fig.colorbar(p4, ax=ax7, ticks=[5, 10.5, 15, 25, 35, 45])
    cbar.ax.set_yticklabels(['Background', 'Aerosol', 'Elevated aerosol',
                             'Precipitation', 'Clouds', 'Undefined'])

    for i, ax, lab in zip([20, 30], [ax2, ax4],
                          ['Precipitation', 'Cloud']):
        ax.set_ylabel(lab)
        if (classifier == i).any():
            depo = depo_save[classifier == i]
            depo = depo[depo < 0.8]
            depo = depo[depo > -0.25]
            ax.hist(depo, bins=40)

    bin_time = np.arange(0, 24+0.35, 0.25)
    bin_height = np.arange(0, df.data['range'].max() + 31, 30)

    mask_elevated_aerosol = classifier == 11
    if (mask_elevated_aerosol).any():
        bin_time1h = np.arange(0, 24+0.5, 0.5)
        co, _, _, _ = binned_statistic_2d(range_save[mask_elevated_aerosol],
                                          time_save[mask_elevated_aerosol],
                                          co_save[mask_elevated_aerosol],
                                          bins=[bin_height, bin_time1h],
                                          statistic=np.nanmean)
        cross, _, _, _ = binned_statistic_2d(range_save[mask_elevated_aerosol],
                                             time_save[mask_elevated_aerosol],
                                             cross_save[mask_elevated_aerosol],
                                             bins=[bin_height, bin_time1h],
                                             statistic=np.nanmean)
        depo = (cross-1)/(co-1)
        depo = depo[depo < 0.8]
        depo = depo[depo > -0.25]
        ax6.hist(depo, bins=40)
    ax6.set_ylabel('Elevated_aerosol')
    ax8.set_ylabel('Aerosol')

    mask_aerosol = classifier == 10
    if (mask_aerosol).any():
        bin_time1h = np.arange(0, 24+0.5, 0.5)
        co, _, _, _ = binned_statistic_2d(range_save[mask_aerosol],
                                          time_save[mask_aerosol],
                                          co_save[mask_aerosol],
                                          bins=[bin_height, bin_time1h],
                                          statistic=np.nanmean)
        cross, _, _, _ = binned_statistic_2d(range_save[mask_aerosol],
                                             time_save[mask_aerosol],
                                             cross_save[mask_aerosol],
                                             bins=[bin_height, bin_time1h],
                                             statistic=np.nanmean)
        depo = (cross-1)/(co-1)
        depo = depo[depo < 0.8]
        depo = depo[depo > -0.25]
        ax8.hist(depo, bins=40)

    ax8.set_xlabel('Depolarization ratio', weight='bold')
    ax7.set_xlabel('Time UTC [hour]', weight='bold')

    fig.tight_layout()
    fig.savefig(classifier_folder + '/' + df.filename + '_hist.png',
                dpi=150, bbox_inches='tight')
    plt.close('all')
