import halo_data as hd
import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import pandas as pd
import xarray as xr

# %%
save_folder = r'F:\halo\paper\figures\background_correction_all/32/'
file_list = glob.glob(r'F:\halo\classifier_new\32/*.nc')
for file in file_list[583:]:
    print(file)
    df = xr.open_dataset(file)
    df = hd.bleed_through(df)  # just to get the bleed through mean and std
    _, index = np.unique(df.coords['time'], return_index=True)
    df = df.isel(time=index)
    df = df.loc[{'time': sorted(df.coords['time'].values)}]

    filter_aerosol = df.classified == 10
    wavelet = 'bior2.6'
    avg = df[['co_signal', 'cross_signal']].resample(time='60min').mean(dim='time')
    avg['aerosol_percentage'] = filter_aerosol.resample(time='60min').mean(dim='time')
    threshold_n = np.nanmedian(df[['co_signal']].resample(time='60min').count()['co_signal'].values)
    threshold = df.attrs['background_snr_sd']/np.sqrt(threshold_n)
    for t in avg['time']:
        range_aerosol = avg['aerosol_percentage'].loc[t, :].values > 0.8

        if np.sum(range_aerosol) == 0:
            continue

        co_mean_profile = avg['co_signal'].loc[t, :].values
        cross_mean_profile = avg['cross_signal'].loc[t, :].values

        if all(np.isnan(co_mean_profile)) | all(np.isnan(cross_mean_profile)):
            continue

        background = hd.background_detection(co_mean_profile, threshold)

        if np.sum(background)/len(background) < 0.5:
            continue
        co_corrected, co_corrected_background, co_fitted = hd.background_correction(
            co_mean_profile, background)
        cross_corrected, cross_corrected_background, cross_fitted = hd.background_correction(
            cross_mean_profile, background)

        cross_sd_background = np.nanstd(cross_corrected_background)
        co_sd_background = np.nanstd(co_corrected_background)

        cross_corrected = (cross_corrected - 1) - \
            df.attrs['bleed_through_mean'] * (co_corrected - 1) + 1

        cross_sd_background_bleed = np.sqrt(
            cross_sd_background**2 +
            ((df.attrs['bleed_through_mean'] * (co_corrected - 1))**2 *
             ((df.attrs['bleed_through_sd']/df.attrs['bleed_through_mean'])**2 +
              (co_sd_background/(co_corrected - 1))**2))
        )

        depo_corrected_wave = (cross_corrected - 1) / \
            (co_corrected - 1)

        depo_corrected_sd_wave = np.sqrt(
            (depo_corrected_wave)**2 *
            (
                (cross_sd_background_bleed/(cross_corrected - 1))**2 +
                (co_sd_background/(co_corrected - 1))**2
            ))
        if (np.max(np.abs(co_fitted - 1)) > 4*threshold) | (np.max(np.abs(cross_fitted - 1)) > 4*threshold):
            good_profile_label = 'Bad'
            good_profile = np.repeat(False, len(depo_corrected_wave[range_aerosol]))
        else:
            good_profile = np.repeat(True, len(depo_corrected_wave[range_aerosol]))
            good_profile_label = 'Good'
        result = pd.DataFrame.from_dict({
            'time': t.values,
            'range': avg['range'][range_aerosol].values,
            # original depo value
            'depo': ((cross_mean_profile - 1)/(co_mean_profile - 1))[range_aerosol],
            'depo_wave': depo_corrected_wave[range_aerosol],  # wave depo value
            'depo_wave_sd': depo_corrected_sd_wave[range_aerosol],
            'co_corrected': co_corrected[range_aerosol],
            'good_profile': good_profile
        })

        fig, ax = plt.subplots(1, 3, figsize=(12, 6))
        ax[0].scatter(cross_mean_profile, avg['range'],
                      alpha=0.3, label='cross_signal', c='blue', edgecolors='none', s=20)
        ax[0].plot(cross_fitted, avg['range'], c='blue', label='wavelet-fit cross')
        ax[0].plot(cross_mean_profile[background], avg['range'][background], 'x',
                   alpha=0.3, label='background_cross', c='blue')

        ax[0].scatter(co_mean_profile, avg['range'],
                      alpha=0.3, label='co_signal', c='red', edgecolors='none', s=20)
        ax[0].plot(co_fitted, avg['range'], c='red', label='wavelet-fit co')
        ax[0].plot(co_mean_profile[background], avg['range'][background], 'x',
                   alpha=0.3, label='background_co', c='red')
        ax[0].set_xlim([0.9995, 1.003])
        ax[0].set_xlabel('SNR')

        ax[1].plot(cross_corrected, avg['range'], '.',
                   label='corrected cross', c='blue')
        ax[1].plot(co_corrected, avg['range'], '.',
                   label='corrected co', c='red')
        ax[1].set_xlim([0.9995, 1.003])
        ax[1].set_xlabel('SNR')

        ax[2].plot(((cross_mean_profile - 1)/(co_mean_profile - 1))[range_aerosol],
                   avg['range'][range_aerosol], '.',
                   label='original depo')
        ax[2].errorbar(depo_corrected_wave[range_aerosol], avg['range'][range_aerosol],
                       xerr=depo_corrected_sd_wave[range_aerosol],
                       label='corrected depo',
                       errorevery=1, elinewidth=0.5, fmt='.')
        ax[2].set_xlim([-0.05, 0.4])
        ax[2].set_xlabel('Depolarization ratio (only Aerosol) ' + good_profile_label)

        for ax_ in ax.flatten():
            ax_.grid()
            ax_.legend()
            ax_.yaxis.set_major_formatter(hd.m_km_ticks())
            ax_.set_ylabel('Height [km]')

        depo_sub_folder = save_folder + '/' + df.attrs['file_name']
        Path(depo_sub_folder).mkdir(parents=True, exist_ok=True)

        # Append to or create new csv file
        with open(depo_sub_folder + '/' +
                  df.attrs['file_name'] + '_aerosol_bkg_corrected.csv', 'a') as f:
            result.to_csv(f, header=f.tell() == 0, index=False)

        fig.savefig(depo_sub_folder + '/' + df.attrs['file_name'] +
                    '_aerosol_bkg_corrected_' + str(pd.to_datetime(t.values).hour),
                    bbox_inches='tight')
        plt.close('all')
