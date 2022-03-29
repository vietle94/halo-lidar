import halo_data as hd
import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import pandas as pd
import xarray as xr
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import r2_score
import statsmodels.graphics.tsaplots as tsa
import statsmodels.stats.diagnostic as diag
%matplotlib qt

# %%
file = r'F:\halo\classifier_new\32/2018-02-12-Uto-32_classified.nc'
df = xr.open_dataset(file)
df = hd.bleed_through(df)  # just to get the bleed through mean and std
_, index = np.unique(df.coords['time'], return_index=True)
df = df.isel(time=index)
df = df.loc[{'time': sorted(df.coords['time'].values)}]

filter_aerosol = df.classified == 10
wavelet = 'bior2.6'
avg = df[['co_signal', 'cross_signal']].resample(time='60min').mean(dim='time')
avg['aerosol_percentage'] = filter_aerosol.resample(time='60min').mean(dim='time')
threshold_n = np.nanmedian(df[['co_signal']].resample(
    time='60min').count()['co_signal'].values)
threshold = df.attrs['background_snr_sd']/np.sqrt(threshold_n)

t = [x for x in avg['time'] if pd.Timestamp(x.values).hour == 0][0]

range_aerosol = avg['aerosol_percentage'].loc[t, :].values > 0.8
co_mean_profile = avg['co_signal'].loc[t, :].values
cross_mean_profile = avg['cross_signal'].loc[t, :].values
background = hd.background_detection(co_mean_profile, threshold)

co_corrected, co_corrected_background, co_fitted = hd.background_correction(
    co_mean_profile, background)
cross_corrected, cross_corrected_background, cross_fitted = hd.background_correction(
    cross_mean_profile, background)

# %%
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(avg['range'], cross_mean_profile,
           alpha=0.3, label='cross_signal', c='blue', edgecolors='none', s=20)
ax.plot(avg['range'], cross_fitted, c='blue', label='wavelet-fit cross')
ax.plot(avg['range'][background], cross_mean_profile[background], 'x',
        alpha=0.3, label='background_cross', c='blue')

ax.scatter(avg['range'], co_mean_profile,
           alpha=0.3, label='co_signal', c='red', edgecolors='none', s=20)
ax.plot(avg['range'], co_fitted, c='red', label='wavelet-fit co')
ax.plot(avg['range'][background], co_mean_profile[background], 'x',
        alpha=0.3, label='background_co', c='red')
ax.set_ylim([0.9995, 1.0009])
ax.set_ylabel('SNR')


ax.grid()
ax.legend()
ax.xaxis.set_major_formatter(hd.m_km_ticks())
ax.set_xlabel('Height [km]')
text = f"$R^2 = {r2_score(co_mean_profile[background], co_fitted[background]):0.3f}$"
ax.set_title(df.file_name + ' at ' + str(pd.Timestamp(t.values).time()) +
             '\n' + text)


# %%
tsa.plot_acf(co_mean_profile[background], lags=40, alpha=0.05,
             title='Auto-correlation coefficients for lags 1 through 40')

# %%
result1 = diag.acorr_ljungbox(co_mean_profile[background], lags=[40], boxpierce=True)
result2 = adfuller(co_mean_profile[background])
print(f'Ljung-Box p-value: {result1[1][0]}')
print(f'Box-Pierce p-value: {result1[3][0]}')
print('ADF p-value: %f' % result2[1])
