from scipy.ndimage import median_filter
from scipy.ndimage import maximum_filter
import numpy as np
import halo_data as hd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl
from scipy.stats import binned_statistic_2d
import pandas as pd


data = hd.getdata('F:/halo/46/depolarization')
classifier_folder = 'F:\\halo\\classifier'
Path(classifier_folder).mkdir(parents=True, exist_ok=True)

date = '2018'
files = [file for file in data if date in file]
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
    v_save = df.data['v_raw'].flatten()
    depo_save = df.data['depo_adj'].flatten()
    co_save = df.data['co_signal'].flatten()
    cross_save = df.data['cross_signal'].flatten()  # Already adjusted with bleed

    df.data['classifier'] = np.zeros(df.data['beta_raw'].shape, dtype=int)

    # Aerosol
    aerosol = df.decision_tree(depo_thres=[None, None],
                               beta_thres=[None, -5.5],
                               v_thres=[None, None],
                               depo=df.data['depo_adj'],
                               beta=np.log10(df.data['beta_raw']),
                               v=df.data['v_raw'])

    # Small size median filter to remove noise
    aerosol_smoothed = median_filter(aerosol, size=11)
    df.data['classifier'][aerosol_smoothed] = 1

    df.filter(variables=['beta_raw', 'v_raw', 'depo_adj'],
              ref='co_signal',
              threshold=1 + 3 * df.snr_sd)

    # Liquid
    liquid = df.decision_tree(depo_thres=[None, None],
                              beta_thres=[-5.5, None],
                              v_thres=[None, None],
                              depo=df.data['depo_adj'],
                              beta=np.log10(df.data['beta_raw']),
                              v=df.data['v_raw'])

    # maximum filter to increase the size of liquid region
    liquid_max = maximum_filter(liquid, size=5)
    # Median filter to remove background noise
    liquid_smoothed = median_filter(liquid_max, size=13)
    # use snr threshold
    snr = df.data['co_signal'] > (1 + 3*df.snr_sd)
    lidquid_smoothed = liquid_smoothed * snr
    df.data['classifier'][liquid_smoothed] = 3

    # updraft - indication of aerosol zone
    updraft = df.decision_tree(depo_thres=[None, None],
                               beta_thres=[None, None],
                               v_thres=[1, None],
                               depo=df.data['depo_adj'],
                               beta=np.log10(df.data['beta_raw']),
                               v=df.data['v_raw'])
    updraft_smooth = median_filter(updraft, size=3)
    updraft_max = maximum_filter(updraft_smooth, size=91)

    # Fill the gap in aerosol zone
    updraft_median = median_filter(updraft_max, size=31)

    # precipitation < -1 (center of precipitation)
    precipitation_1 = df.decision_tree(depo_thres=[None, None],
                                       beta_thres=[-7, None],
                                       v_thres=[None, -1],
                                       depo=df.data['depo_adj'],
                                       beta=np.log10(df.data['beta_raw']),
                                       v=df.data['v_raw'])
    precipitation_1_median = median_filter(precipitation_1, size=9)

    # Only select precipitation outside of aerosol zone
    precipitation_1_ne = precipitation_1_median * ~updraft_median
    precipitation_1_median_smooth = median_filter(precipitation_1_ne,
                                                  size=3)
    precipitation = precipitation_1_median_smooth

    # precipitation < -0.5 (include of precipitation)
    precipitation_1_low = df.decision_tree(depo_thres=[None, None],
                                           beta_thres=[-7, None],
                                           v_thres=[None, -0.5],
                                           depo=df.data['depo_adj'],
                                           beta=np.log10(df.data['beta_raw']),
                                           v=df.data['v_raw'])

    # Avoid ebola infection surrounding updraft
    # Useful to contain error during ebola precipitation
    updraft_ebola = df.decision_tree(depo_thres=[None, None],
                                     beta_thres=[None, None],
                                     v_thres=[0.2, None],
                                     depo=df.data['depo_adj'],
                                     beta=np.log10(df.data['beta_raw']),
                                     v=df.data['v_raw'])
    updraft_ebola_max = maximum_filter(updraft_ebola, size=3)

    # Ebola precipitation
    for _ in range(100):
        prep_1_max = maximum_filter(precipitation, size=3)
        prep_1_max *= ~updraft_ebola_max  # Avoid updraft area
        precipitation = precipitation_1_low * prep_1_max

    df.data['classifier'][precipitation] = 2

    # Remove all aerosol above cloud or precipitation
    mask_aerosol0 = df.data['classifier'] == 1
    for i in np.array([2, 3]):
        mask = df.data['classifier'] == i
        mask_row = np.argwhere(mask.any(axis=1)).reshape(-1)
        mask_col = np.nanargmax(df.data['classifier'][mask_row, :] == i,
                                axis=1)
        for row, col in zip(mask_row, mask_col):
            mask[row, col:] = True
        mask_undefined = mask * mask_aerosol0
        df.data['classifier'][mask_undefined] = i

    fig1, axes = plt.subplots(6, 2, sharex=True, sharey=True,
                              figsize=(16, 9))
    for val, ax, cmap in zip([aerosol, aerosol_smoothed,
                              liquid_smoothed, precipitation_1_median,
                              updraft_median,
                              precipitation_1_median_smooth, precipitation_1_low,
                              updraft_ebola_max, precipitation],
                             axes.flatten()[2:-1],
                             [['white', '#2ca02c'], ['white', '#2ca02c'],
                              ['white', 'red'], ['white', 'blue'],
                              ['white', '#D2691E'],
                              ['white', 'blue'], ['white', 'blue'],
                              ['white', '#D2691E'], ['white', 'blue']]):
        ax.pcolormesh(df.data['time'], df.data['range'],
                      val.T, cmap=mpl.colors.ListedColormap(cmap))
    axes.flatten()[-1].pcolormesh(df.data['time'], df.data['range'],
                                  df.data['classifier'].T,
                                  cmap=mpl.colors.ListedColormap(
        ['white', '#2ca02c', 'blue', 'red']),
        vmin=0, vmax=3)
    axes[0, 0].pcolormesh(df.data['time'], df.data['range'],
                          np.log10(df.data['beta_raw']).T,
                          cmap='jet', vmin=-8, vmax=-4)
    axes[0, 1].pcolormesh(df.data['time'], df.data['range'],
                          df.data['v_raw'].T, cmap='jet', vmin=-2, vmax=2)
    fig1.tight_layout()
    fig1.savefig(classifier_folder + '/' + df.filename + '_classifier.png',
                 dpi=150, bbox_inches='tight')
    plt.close(fig1)

    # %%
    classifier = df.data['classifier'].flatten()
    time_save = np.repeat(df.data['time'],
                          df.data['beta_raw'].shape[1])
    range_save = np.tile(df.data['range'],
                         df.data['beta_raw'].shape[0])

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
    ax1 = fig.add_subplot(421)
    ax2 = fig.add_subplot(422)
    ax3 = fig.add_subplot(423, sharex=ax1)
    ax4 = fig.add_subplot(424, sharex=ax2)
    ax5 = fig.add_subplot(425, sharex=ax1)
    ax6 = fig.add_subplot(426, sharex=ax2)
    ax7 = fig.add_subplot(427, sharex=ax1)
    ax8 = fig.add_subplot(428, sharex=ax2)
    ax1.pcolormesh(df.data['time'], df.data['range'],
                   np.log10(df.data['beta_raw']).T, cmap='jet', vmin=-8, vmax=-4)
    ax3.pcolormesh(df.data['time'], df.data['range'],
                   df.data['v_raw'].T, cmap='jet', vmin=-2, vmax=2)
    ax5.pcolormesh(df.data['time'], df.data['range'],
                   df.data['depo_adj'].T, cmap='jet', vmin=0, vmax=0.5)
    ax7.pcolormesh(df.data['time'], df.data['range'],
                   df.data['classifier'].T, vmin=0, vmax=3,
                   cmap=mpl.colors.ListedColormap(
        ['white', '#2ca02c', 'blue', 'red']))

    bin_time = np.arange(0, 24+0.35, 0.25)
    bin_height = np.arange(0, df.data['range'].max() + 31, 30)
    for i, ax, lab in zip([1, 2, 3], [ax2, ax4, ax6],
                          ['aerosol_15min', 'precipitation', 'clouds']):
        ax.set_ylabel(lab)
        if (classifier == i).any():
            co, _, _, _ = binned_statistic_2d(range_save[classifier == i],
                                              time_save[classifier == i],
                                              co_save[classifier == i],
                                              bins=[bin_height, bin_time],
                                              statistic=np.nanmean)
            cross, _, _, _ = binned_statistic_2d(range_save[classifier == i],
                                                 time_save[classifier == i],
                                                 cross_save[classifier == i],
                                                 bins=[bin_height, bin_time],
                                                 statistic=np.nanmean)
            depo = (cross-1)/(co-1)
            depo = depo[depo < 0.8]
            depo = depo[depo > -0.25]
            ax.hist(depo, bins=40)

    ax8.set_ylabel('aerosol_1hr')
    if (classifier == 1).any():
        bin_time1h = np.arange(0, 24+1.5, 1)
        co, _, _, _ = binned_statistic_2d(range_save[classifier == 1],
                                          time_save[classifier == 1],
                                          co_save[classifier == 1],
                                          bins=[bin_height, bin_time1h],
                                          statistic=np.nanmean)
        cross, _, _, _ = binned_statistic_2d(range_save[classifier == 1],
                                             time_save[classifier == 1],
                                             cross_save[classifier == 1],
                                             bins=[bin_height, bin_time1h],
                                             statistic=np.nanmean)
        depo = (cross-1)/(co-1)
        depo = depo[depo < 0.8]
        depo = depo[depo > -0.25]
        ax.hist(depo, bins=40)

    bin_time1h = np.arange(0, 24+1.5, 1)
    co, _, _, _ = binned_statistic_2d(range_save[classifier == 1],
                                      time_save[classifier == 1],
                                      co_save[classifier == 1],
                                      bins=[bin_height, bin_time1h],
                                      statistic=np.nanmean)
    cross, _, _, _ = binned_statistic_2d(range_save[classifier == 1],
                                         time_save[classifier == 1],
                                         cross_save[classifier == 1],
                                         bins=[bin_height, bin_time1h],
                                         statistic=np.nanmean)
    depo = (cross-1)/(co-1)
    depo = depo[depo < 0.8]
    depo = depo[depo > -0.25]
    ax8.hist(depo, bins=40)
    ax8.set_ylabel('aerosol_1hr')

    fig.tight_layout()
    fig.savefig(classifier_folder + '/' + df.filename + '_hist.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
