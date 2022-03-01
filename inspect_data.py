from sklearn.metrics import r2_score
import halo_data as hd
import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import pandas as pd
from matplotlib.widgets import RectangleSelector, SpanSelector
from matplotlib.ticker import FuncFormatter
from matplotlib.widgets import Button
import xarray as xr
import json
from scipy.signal import find_peaks, peak_widths
from matplotlib.ticker import FormatStrFormatter

import pywt
%matplotlib qt


def bleed_through(df):
    # Correction for bleed through and remove all observations below 90m
    with open('summary_info.json', 'r') as file:
        summary_info = json.load(file)
    df = df.where(df.range > 90, drop=True)
    file_date = '-'.join([str(int(df.attrs[ele])).zfill(2) for
                          ele in ['year', 'month', 'day']])
    file_location = '-'.join([df.attrs['location'], str(int(df.attrs['systemID']))])
    df.attrs['file_name'] = file_date + '-' + file_location

    if '32' in file_location:
        for period in summary_info['32']:
            if (period['start_date'] <= file_date) & \
                    (file_date <= period['end_date']):
                df.attrs['background_snr_sd'] = period['snr_sd']
                df.attrs['bleed_through_mean'] = period['bleed_through']['mean']
                df.attrs['bleed_through_sd'] = period['bleed_through']['sd']
    else:
        id = str(int(df.attrs['systemID']))
        df.attrs['background_snr_sd'] = summary_info[id]['snr_sd']
        df.attrs['bleed_through_mean'] = summary_info[id]['bleed_through']['mean']
        df.attrs['bleed_through_sd'] = summary_info[id]['bleed_through']['sd']
    bleed = df.attrs['bleed_through_mean']
    sigma_bleed = df.attrs['bleed_through_sd']
    sigma_co, sigma_cross = df.attrs['background_snr_sd'], df.attrs['background_snr_sd']

    df['cross_signal_bleed'] = (['time', 'range'], ((df['cross_signal'] - 1) -
                                                    bleed * (df['co_signal'] - 1) + 1).data)

    df['cross_signal_bleed_sd'] = np.sqrt(
        sigma_cross**2 +
        ((bleed * (df['co_signal'] - 1))**2 *
         ((sigma_bleed/bleed)**2 +
          (sigma_co/(df['co_signal'] - 1))**2))
    )
    df['depo_bleed'] = (df['cross_signal_bleed'] - 1) / \
        (df['co_signal'] - 1)

    df['depo_bleed_sd'] = np.sqrt(
        (df['depo_bleed'])**2 *
        (
            (df['cross_signal_bleed_sd']/(df['cross_signal_bleed'] - 1))**2 +
            (sigma_co/(df['co_signal']-1))**2
        ))
    return df

# %%


class span_select():

    def __init__(self, x, y, ax_in, canvas, orient='vertical'):
        self.x, self.y = x, y
        self.ax_in = ax_in
        self.canvas = canvas
        self.selector = SpanSelector(
            self.ax_in, self, orient, span_stays=True, useblit=True
        )

    def __call__(self, min, max):
        self.masky = (self.y > min) & (self.y < max)
        self.selected_x = self.x[self.masky]
        self.selected_y = self.y[self.masky]


class span_aerosol(span_select):

    def __init__(self, x, y, ax_in, canvas, orient, cross, range_aerosol,
                 ax2, ax3, ax4, bleed_mean, bleed_sd):
        super().__init__(x, y, ax_in, canvas, orient)
        self.range_aerosol = range_aerosol
        self.co, self.range, self.cross = x, y, cross
        self.co_all = np.array([])
        self.cross_all = np.array([])
        self.range_all = np.array([])
        self.ax2 = ax2
        self.ax3, self.ax4 = ax3, ax4
        self.bleed_mean = bleed_mean
        self.bleed_sd = bleed_sd

    def __call__(self, min, max):
        self.min = min
        self.max = max
        self.maskrange = (self.range > self.min) & (self.range < self.max)
        self.selected_co = self.co[self.maskrange]
        self.selected_range = self.range[self.maskrange]
        self.selected_cross = self.cross[self.maskrange]

    def apply(self, event):
        self.co_all = np.append(self.co_all, self.selected_co)
        self.cross_all = np.append(self.cross_all, self.selected_cross)
        self.range_all = np.append(self.range_all, self.selected_range)
        self.ax_in.axhspan(self.min, self.max, alpha=0.5, color='yellow')
        self.canvas.draw()
        # print(f'you choose the background area between: \n' +
        #       f'{self.min/1000:.2f}km and {self.max/1000:.2}km')

    def fit(self, event):
        co = self.co_all
        cross = self.cross_all
        x_co = self.range_all

        a, b, c = np.polyfit(x_co, co, deg=2)
        y_co = c + b*self.range + a*(self.range**2)
        y_co_background = c + b*x_co + a*(x_co**2)

        a, b, c = np.polyfit(x_co, cross, deg=2)
        y_cross = c + b*self.range + a*(self.range**2)
        y_cross_background = c + b*x_co + a*(x_co**2)

        self.ax_in.plot(y_co, self.range, c='red', label='eye-fit co')
        self.ax_in.plot(y_cross, self.range, c='blue', label='eye-fit cross')
        self.ax_in.legend()
        co_corrected = self.co/y_co
        self.co_corrected = co_corrected
        cross_corrected = self.cross/y_cross
        cross_sd_background = np.nanstd(cross/y_cross_background)
        co_sd_background = np.nanstd(co/y_co_background)
        self.sigma_co, self.sigma_cross = co_sd_background, cross_sd_background
        bleed = self.bleed_mean
        sigma_bleed = self.bleed_sd

        self.cross_corrected = (cross_corrected - 1) - \
            bleed * (co_corrected - 1) + 1

        cross_sd_background_bleed = np.sqrt(
            self.sigma_cross**2 +
            ((bleed * (co_corrected - 1))**2 *
             ((sigma_bleed/bleed)**2 +
              (self.sigma_co/(co_corrected - 1))**2))
        )

        self.depo_corrected = (cross_corrected - 1) / \
            (co_corrected - 1)

        self.depo_corrected_sd = np.sqrt(
            (self.depo_corrected)**2 *
            (
                (cross_sd_background_bleed/(cross_corrected - 1))**2 +
                (self.sigma_co/(co_corrected - 1))**2
            ))

        # self.ax3.plot(((self.cross - 1)/(self.co - 1))[self.range_aerosol],
        #               self.range[self.range_aerosol], '.',
        #               label='Original depo')
        # self.ax3.legend()
        # self.ax3.set_xlim([-0.1, 0.4])
        # self.ax3.set_xlabel(r'$\delta$')
        # self.ax3.set_ylabel('Height [km]')
        self.ax3.yaxis.set_major_formatter(m_km_ticks())
        self.ax4.errorbar(self.depo_corrected[self.range_aerosol],
                          self.range[self.range_aerosol],
                          xerr=self.depo_corrected_sd[self.range_aerosol],
                          errorevery=1, elinewidth=0.5, fmt='.', label='Eye-corrected depo')

        # Wavelet
        # coeff = pywt.swt(np.pad(self.co-1, (0, 67), 'constant',
        #                         constant_values=(0, 0)),
        #                  'bior2.6', level=5)
        # uthresh = np.median(np.abs(coeff[1]))/0.6745 * np.sqrt(2 * np.log(len(coeff[1])))
        # coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
        # filtered = pywt.iswt(coeff, wavelet) + 1
        # filtered = filtered[:len(self.co)]
        # background = filtered < 1+6e-5
        #
        # selected_range = self.range[background]
        # selected_co = self.co[background]
        # selected_cross = self.cross[background]
        #
        # a, b, c = np.polyfit(selected_range, selected_co, deg=2)
        # y_co = c + b*self.range + a*(self.range**2)
        # y_co_background = c + b*selected_range + a*(selected_range**2)
        #
        # a, b, c = np.polyfit(selected_range, selected_cross, deg=2)
        # y_cross = c + b*self.range + a*(self.range**2)
        # y_cross_background = c + b*selected_range + a*(selected_range**2)

        co_corrected, co_corrected_background, co_fitted = background_correction(self.co)
        cross_corrected, cross_corrected_background, cross_fitted = background_correction(
            self.cross)

        self.ax2.plot(co_fitted, self.range, c='red', label='wavelet-fit co')
        self.ax2.plot(cross_fitted, self.range, c='blue', label='wavelet-fit cross')
        self.ax2.legend()
        # co_corrected = self.co/y_co
        self.co_corrected = co_corrected
        # cross_corrected = self.cross/y_cross
        cross_sd_background = np.nanstd(cross_corrected_background)
        co_sd_background = np.nanstd(co_corrected_background)
        self.sigma_co, self.sigma_cross = co_sd_background, cross_sd_background
        bleed = self.bleed_mean
        sigma_bleed = self.bleed_sd

        self.cross_corrected = (cross_corrected - 1) - \
            bleed * (co_corrected - 1) + 1

        cross_sd_background_bleed = np.sqrt(
            self.sigma_cross**2 +
            ((bleed * (co_corrected - 1))**2 *
             ((sigma_bleed/bleed)**2 +
              (self.sigma_co/(co_corrected - 1))**2))
        )

        self.depo_corrected_wave = (cross_corrected - 1) / \
            (co_corrected - 1)

        self.depo_corrected_sd_wave = np.sqrt(
            (self.depo_corrected_wave)**2 *
            (
                (cross_sd_background_bleed/(cross_corrected - 1))**2 +
                (self.sigma_co/(co_corrected - 1))**2
            ))

        self.ax4.errorbar(self.depo_corrected_wave[self.range_aerosol],
                          self.range[self.range_aerosol],
                          xerr=self.depo_corrected_sd_wave[self.range_aerosol],
                          errorevery=1, elinewidth=0.5, fmt='.',
                          label='Wavelet-corrected depo')
        self.ax4.plot(((self.cross - 1)/(self.co - 1))[self.range_aerosol],
                      self.range[self.range_aerosol], 'x',
                      label='Original depo')

        self.ax4.legend()
        self.ax4.set_ylabel('Height [km]')
        self.ax4.yaxis.set_major_formatter(m_km_ticks())
        self.ax4.set_xlabel(r'$\delta$')

        self.ax3.errorbar(self.depo_corrected_wave[self.range_aerosol],
                          self.depo_corrected[self.range_aerosol],
                          xerr=self.depo_corrected_sd_wave[self.range_aerosol],
                          yerr=self.depo_corrected_sd[self.range_aerosol],
                          errorevery=1, elinewidth=0.5, fmt='.')
        self.ax3.set_xlabel('Wavelet corrected')
        self.ax3.set_ylabel('Eye-corrected')
        self.ax3.set_xlim([0, 0.35])
        self.ax3.set_ylim([0, 0.35])
        self.ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        z = np.polyfit(self.depo_corrected_wave[self.range_aerosol],
                       self.depo_corrected[self.range_aerosol], 1)
        y_hat = np.poly1d(z)(self.depo_corrected_wave[self.range_aerosol])

        text = f"$y={z[0]:0.3f}\;x{z[1]:+0.3f}$\n$R^2 = {r2_score(self.depo_corrected[self.range_aerosol],y_hat):0.3f}$"
        self.ax3.text(0.05, 0.95, text, transform=self.ax3.transAxes,
                      fontsize=10, verticalalignment='top')

        self.ax3.axline((0, 0), (0.35, 0.35), color='grey', linewidth=0.5,
                        ls='--')
        self.canvas.draw()


# %%
class area_aerosol():

    def __init__(self, fig, ax1, ax2, ax3, ax4, avg, bleed_mean, bleed_sd,
                 save_path, t):
        self.save_path = save_path
        self.co = avg['co_signal']
        self.cross = avg['cross_signal']
        self.range = avg['range']
        self.aerosol_percentage = avg['aerosol_percentage']
        self.ax1 = ax1
        self.ax2, self.ax3, self.ax4 = ax2, ax3, ax4
        self.canvas = fig.canvas
        self.fig = fig
        self.bleed_mean = bleed_mean
        self.bleed_sd = bleed_sd
        self.axnext = self.fig.add_axes([0.59, 0.005, 0.1, 0.025])
        self.bnext = Button(self.axnext, 'Next')
        self.bnext.on_clicked(self.myf)
        self.axsave = self.fig.add_axes([0.48, 0.005, 0.1, 0.025])
        self.bsave = Button(self.axsave, 'Save')
        self.bsave.on_clicked(self.save)
        self.axapply = self.fig.add_axes([0.7, 0.005, 0.1, 0.025])
        self.axfit = self.fig.add_axes([0.81, 0.005, 0.1, 0.025])
        self.bfit = Button(self.axfit, 'Fit')
        self.bapply = Button(self.axapply, 'Apply')
        self.t = t

    def myf(self, event):
        self.ax1.cla()
        self.ax2.cla()
        self.ax3.cla()
        self.ax4.cla()
        # self.t += 1
        self.range_aerosol = self.aerosol_percentage[self.t, :] > 0.8
        self.co_mean_profile = self.co[self.t, :]
        self.cross_mean_profile = self.cross[self.t, :]
        self.ax1.plot(self.co_mean_profile,
                      self.range, '.', label='co_signal', c='red')
        self.ax1.plot(self.cross_mean_profile,
                      self.range, '.', label='cross_signal', c='blue')
        self.ax1.yaxis.set_major_formatter(m_km_ticks())
        self.ax1.legend()
        self.ax1.set_xlim([0.9995, 1.003])
        self.ax1.set_ylabel('Height [km]')

        self.ax2.plot(self.co_mean_profile,
                      self.range, '.', label='co_signal', c='red')
        self.ax2.plot(self.cross_mean_profile,
                      self.range, '.', label='cross_signal', c='blue')
        self.ax2.yaxis.set_major_formatter(m_km_ticks())
        self.ax2.legend()
        self.ax2.set_xlim([0.9995, 1.003])
        self.canvas.draw()
        self.span_aerosol = span_aerosol(self.co_mean_profile, self.range, self.ax1,
                                         self.canvas, 'vertical',
                                         self.cross_mean_profile,
                                         self.range_aerosol,
                                         self.ax2, self.ax3, self.ax4,
                                         self.bleed_mean, self.bleed_sd)
        self.bfit.on_clicked(self.span_aerosol.fit)
        self.bapply.on_clicked(self.span_aerosol.apply)

    def save(self, event):
        self.fig.savefig(self.save_path + str(self.t),
                         bbox_inches='tight')


def m_km_ticks():
    '''
    Modify ticks from m to km
    '''
    return FuncFormatter(lambda x, pos: f'{x/1000:.0f}')


def background_correction(x, wavelet='bior2.6'):
    coeff = pywt.swt(np.pad(x-1, (0, (len(x) // 2**5 + 3) * 2**5 - len(x)), 'constant', constant_values=(0, 0)),
                     wavelet, level=5)
    uthresh = np.median(np.abs(coeff[1]))/0.6745 * np.sqrt(2 * np.log(len(coeff[1])))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    filtered = pywt.iswt(coeff, wavelet) + 1
    filtered = filtered[:len(x)]

    peaks, _ = find_peaks(-filtered)
    peaks = peaks[filtered[peaks] < 1]

    indices = np.arange(len(x))
    a, b, c = np.polyfit(peaks, filtered[peaks], deg=2)
    smooth_peaks = c + b*indices + a*(indices**2)
    background = filtered < smooth_peaks + 6e-5

    selected_x = filtered[background]
    selected_indices = indices[background]

    a, b, c = np.polyfit(selected_indices, selected_x, deg=2)
    x_fitted = c + b*indices + a*(indices**2)
    x_background_fitted = c + b*selected_indices + a*(selected_indices**2)

    x_corrected = x/x_fitted
    x_background_corrected = selected_x/x_background_fitted

    return x_corrected, x_background_corrected, x_fitted


# %%
df = xr.open_dataset(r'F:\halo\classifier_new\46/2018-04-15-Hyytiala-46_classified.nc')
df = bleed_through(df)  # just to get the bleed through mean and std

filter_aerosol = df.classified == 10
wavelet = 'bior2.6'
avg = df[['co_signal', 'cross_signal']].resample(time='60min').mean(dim='time')
avg['aerosol_percentage'] = filter_aerosol.resample(time='60min').mean(dim='time')
bleed_mean = df.attrs['bleed_through_mean']
bleed_sd = df.attrs['bleed_through_sd']

# %%
fig = plt.figure(figsize=(16, 9))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222, sharex=ax1, sharey=ax1)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
p = area_aerosol(fig, ax1, ax2, ax3, ax4, avg, bleed_mean, bleed_sd,
                 r'F:\halo\paper\figures\background_compare/compare_', 23)

# %%
fig.savefig(r'F:\halo\paper\figures\background_compare/' + df.attrs['file_name'] + '_background_compare_' + str(p.t),
            bbox_inches='tight')

result = pd.DataFrame.from_dict({
    'time': p.span_aerosol.depo_corrected_wave.time.values,
    'depo': ((p.span_aerosol.cross - 1)/(p.span_aerosol.co - 1)).values,  # original depo value
    'depo_eye': p.span_aerosol.depo_corrected.values,  # eye depo value
    'depo_wave': p.span_aerosol.depo_corrected_wave.values,  # wave depo value
    'range': p.span_aerosol.depo_corrected_wave.range.values,
    'aerosol': p.span_aerosol.range_aerosol.values
})

with open(r'F:\halo\paper\figures\background_compare/' + df.attrs['file_name'] + '_background_compare_' + str(p.t) + '.csv', 'w') as f:
    result.to_csv(f, header=f.tell() == 0, index=False)
