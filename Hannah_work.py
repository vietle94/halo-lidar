from scipy.ndimage import median_filter
from scipy.ndimage import maximum_filter
import numpy as np
import halo_data as hd
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import matplotlib as mpl


data = hd.getdata('F:/halo/46/depolarization')
classifier_folder = 'F:\\halo\\classifier'
Path(classifier_folder).mkdir(parents=True, exist_ok=True)

date = '201806'
files = [file for file in data if date in file]

for file in files:
    df = hd.halo_data(file)
    df.filter_height()
    df.unmask999()
    df.depo_cross_adj()

    df.filter(variables=['beta_raw'],
              ref='co_signal',
              threshold=1 + 2 * df.snr_sd)

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

    # Precipitation < -1.5m/s
    precipitation_15 = df.decision_tree(depo_thres=[None, None],
                                        beta_thres=[-6.5, None],
                                        v_thres=[None, -1.5],
                                        depo=df.data['depo_adj'],
                                        beta=np.log10(df.data['beta_raw']),
                                        v=df.data['v_raw'])

    # precipitation_15_median = median_filter(precipitation_15, size=(9, 33))
    precipitation_15_median_smooth = median_filter(precipitation_15,
                                                   size=(9, 17))
    precipitation_15_median = precipitation_15_median_smooth

    # Precipitation < -1m/s
    precipitation_1 = df.decision_tree(depo_thres=[None, None],
                                       beta_thres=[-7, None],
                                       v_thres=[None, -1],
                                       depo=df.data['depo_adj'],
                                       beta=np.log10(df.data['beta_raw']),
                                       v=df.data['v_raw'])

    # Ebola precipitation
    for _ in range(100):
        prep_15_max = maximum_filter(precipitation_15_median_smooth, size=9)
        precipitation_15_median_smooth = precipitation_1 * prep_15_max

    precipitation = precipitation_15_median_smooth
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

    fig, axes = plt.subplots(6, 2, sharex=True, sharey=True,
                             figsize=(16, 9))
    for val, ax, cmap in zip([aerosol, aerosol_smoothed,
                              liquid, liquid_max, liquid_smoothed,
                              precipitation_15, precipitation_15_median,
                              precipitation_1, precipitation],
                             axes.flatten()[2:-1],
                             [['white', '#2ca02c'], ['white', '#2ca02c'],
                              ['white', 'red'], ['white', 'red'],
                              ['white', 'red'],
                              ['white', 'blue'], ['white', 'blue'],
                              ['white', 'blue'], ['white', 'blue']]):
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
    fig.tight_layout()
    fig.savefig(classifier_folder + '/' + df.filename + '_classifier.png',
                dpi=150, bbox_inches='tight')

    ###############################
    #     Hannah, no need to run any after this line
    ###############################
    classifier = df.data['classifier'].flatten()
    beta_aerosol = df.data['beta_raw'].flatten()
    v_aerosol = df.data['v_raw'].flatten()
    depo_aerosol = df.data['depo_adj'].flatten()
    time_aerosol = np.repeat(df.data['time'],
                             df.data['beta_raw'].shape[1])
    range_aerosol = np.tile(df.data['range'],
                            df.data['beta_raw'].shape[0])

    result = pd.DataFrame({'date': df.date,
                           'location': df.location,
                           'beta': np.log10(beta_aerosol),
                           'v': v_aerosol,
                           'depo': depo_aerosol,
                           'time': time_aerosol,
                           'range': range_aerosol,
                           'classifier': classifier})

    result.to_csv(classifier_folder + '/' + df.filename + '_classified.csv',
                  index=False)

    temp = result[['depo', 'classifier']]
    temp = temp[(temp['depo'] < 0.6) & (temp['depo'] > -0.25)]
    fig = plt.figure(figsize=(16, 9))
    ax1 = fig.add_subplot(421)
    ax2 = fig.add_subplot(422)
    ax3 = fig.add_subplot(423, sharex=ax1)
    ax4 = fig.add_subplot(424, sharex=ax2)
    ax5 = fig.add_subplot(425, sharex=ax1)
    ax6 = fig.add_subplot(426, sharex=ax2)
    ax7 = fig.add_subplot(414)
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
    ax2.hist(temp.loc[temp['classifier'] == 1, 'depo'], bins=40)
    ax2.set_ylabel('aerosol')
    ax4.hist(temp.loc[temp['classifier'] == 2, 'depo'], bins=40)
    ax4.set_ylabel('precipitation')
    ax6.hist(temp.loc[temp['classifier'] == 3, 'depo'], bins=40)
    ax6.set_ylabel('clouds')
    fig.tight_layout()
    fig.savefig(classifier_folder + '/' + df.filename + '_hist.png',
                dpi=150, bbox_inches='tight')
