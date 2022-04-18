import halo_data as hd
import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import pandas as pd
import xarray as xr
import pystan
import pickle
%matplotlib qt
# %%


def standard_scale(x):
    return (x - np.mean(x))/np.std(x)


def standard_scale_reverse(a, b, c, x0, y0):
    meanx = np.mean(x0)
    stdx = np.std(x0)
    meany = np.mean(y0)
    stdy = np.std(y0)

    a0 = meany + stdy*a - stdy/stdx*b*meanx + stdy*c/((stdx)**2)*(meanx**2)
    b0 = stdy/stdx*b - 2*meanx*stdy*c/((stdx)**2)
    c0 = stdy*c/((stdx)**2)
    # e0 = stdy*e
    return a0, b0, c0


def SNR_fit(SNR, background, model):
    indices = np.arange(len(SNR))
    N = sum(background)
    x_standardized = standard_scale(indices[background])
    y_standardized = standard_scale(SNR[background])
    data = dict(N=N, x=x_standardized, y=y_standardized)
    fit = model.sampling(data=data, warmup=500, iter=2000, chains=4, thin=1)

    sigma = fit['sigma'].copy()
    a0, b0, c0 = standard_scale_reverse(fit['a'], fit['b'],
                                        fit['c'], indices[background],
                                        SNR[background])

    SNR_fit = np.empty((a0.size, indices.size))
    for i in range(a0.size):
        mu = a0[i] + b0[i]*indices + c0[i]*indices*indices
        SNR_fit[i, :] = mu + \
            np.std(SNR[background]) * \
            np.random.normal(0,
                             scale=sigma[i], size=indices.size)
    SNR_corrected = SNR/SNR_fit
    return fit, SNR_fit, SNR_corrected


# %%
background_correction_model = """
data{
    int < lower = 0 > N; // number of background observations
    real y[N]; // SNR
    real x[N]; // range gate
}
parameters{
real a;
real b;
real c;
real <lower = 0> sigma2;
}
transformed parameters{
real mu[N];
real <lower=0> sigma;
sigma = sqrt(sigma2);

for(i in 1: N)
mu[i] = a + b * x[i] + c * x[i] * x[i];
}
model{
a ~ normal(0, sqrt(1e6));
b ~ normal(0, sqrt(1e6));
c ~ normal(0, sqrt(1e6));
sigma2 ~inv_gamma(0.001, 0.001);

for(i in 1: N)
y[i] ~normal(mu[i], sigma);
}
"""

# %%
# model = pystan.StanModel(model_code=background_correction_model)
# with open('background_correction_model.pkl', 'wb') as f:
#     pickle.dump(model, f)

# %%
save_folder = r'F:\halo\paper\figures\background_correction_all/46/'
file_list = glob.glob(r'F:\halo\classifier_new\46/*.nc')
for file in [file_list[1]]:
    file = r'F:\halo\classifier_new\46/2018-07-27-Hyytiala-46_classified.nc'
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
        print(t.values)
        range_aerosol = avg['aerosol_percentage'].loc[t, :].values > 0.8
        co_mean_profile = avg['co_signal'].loc[t, :].values
        cross_mean_profile = avg['cross_signal'].loc[t, :].values
        if all(np.isnan(co_mean_profile)) | all(np.isnan(cross_mean_profile)):
            continue
        background = hd.background_detection(co_mean_profile, threshold)
        if np.sum(background)/len(background) < 0.5:
            continue

        model = pickle.load(open('background_correction_model.pkl', 'rb'))
        print('loaded model')
        _, co_fitted, co_corrected = SNR_fit(co_mean_profile, background, model)
        _, cross_fitted, cross_corrected = SNR_fit(cross_mean_profile, background,
                                                   model)
        print('fitted model')

        bleed = np.random.normal(df.attrs['bleed_through_mean'],
                                 df.attrs['bleed_through_sd'], cross_corrected.shape)

        cross_corrected_bleed = (cross_corrected - 1) - \
            bleed * (co_corrected - 1) + 1

        depo_corrected = (cross_corrected_bleed - 1) / (co_corrected - 1)
        depo = (cross_mean_profile - 1)/(co_mean_profile - 1)

        fig, ax = plt.subplots(1, 4, figsize=(12, 6))
        ax[0].scatter(cross_mean_profile, avg['range'],
                      alpha=0.3, label='cross_signal', c='blue', edgecolors='none', s=20)
        ax[0].plot(np.mean(cross_fitted, axis=0), avg['range'], c='blue', label='wavelet-fit cross')
        ax[0].plot(cross_mean_profile[background], avg['range'][background], 'x',
                   alpha=0.3, label='background_cross', c='blue')

        ax[0].scatter(co_mean_profile, avg['range'],
                      alpha=0.3, label='co_signal', c='red', edgecolors='none', s=20)
        ax[0].plot(np.mean(co_fitted, axis=0), avg['range'], c='red', label='wavelet-fit co')
        ax[0].plot(co_mean_profile[background], avg['range'][background], 'x',
                   alpha=0.3, label='background_co', c='red')
        ax[0].set_xlim([0.9995, 1.003])
        ax[0].set_xlabel('SNR')

        ax[1].errorbar(np.mean(cross_corrected, axis=0), avg['range'],
                       xerr=np.std(cross_corrected, axis=0), fmt='.',
                       label='corrected cross', errorevery=1, elinewidth=0.5,
                       c='blue', alpha=0.5)
        ax[1].set_xlabel('SNR')
        ax[1].set_xlim([0.9995, 1.003])

        ax[2].errorbar(np.mean(co_corrected, axis=0), avg['range'],
                       xerr=np.std(co_corrected, axis=0), fmt='.',
                       label='corrected co', errorevery=1, elinewidth=0.5,
                       color='red', alpha=0.5)
        ax[2].set_xlim([0.9995, 1.003])
        ax[2].set_xlabel('SNR')

        ax[3].plot(depo[range_aerosol],
                   avg['range'][range_aerosol], '.',
                   label='original depo')
        ax[3].errorbar(x=np.mean(depo_corrected, axis=0)[range_aerosol],
                       y=avg['range'][range_aerosol],
                       xerr=np.std(depo_corrected, axis=0)[range_aerosol],
                       label='corrected depo',
                       errorevery=1, elinewidth=0.5, fmt='.')
        ax[3].set_xlim([-0.05, 0.4])
        ax[3].set_xlabel('Depolarization ratio (only Aerosol)')

        for ax_ in ax.flatten():
            ax_.grid()
            ax_.legend()
            ax_.yaxis.set_major_formatter(hd.m_km_ticks())
            ax_.set_ylabel('Height [km]')

        result = pd.DataFrame.from_dict({
            'time': t.values,
            'range': avg['range'][range_aerosol].values,
            # original depo value
            'depo': depo[range_aerosol],
            'depo_corrected': np.mean(depo_corrected, axis=0)[range_aerosol],
            'depo_corrected_sd': np.std(depo_corrected, axis=0)[range_aerosol],
            'co_corrected': np.mean(co_corrected, axis=0)[range_aerosol],
            'co_corrected_sd': np.std(co_corrected, axis=0)[range_aerosol],
            'cross_corrected': np.mean(cross_corrected_bleed, axis=0)[range_aerosol],
            'cross_corrected_sd': np.std(cross_corrected_bleed, axis=0)[range_aerosol]
        })
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
        print('done')
