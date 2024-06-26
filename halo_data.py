import pywt
from scipy.io import netcdf
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector, SpanSelector
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import pandas as pd
from pathlib import Path
from collections import OrderedDict
from matplotlib.colors import LogNorm
from matplotlib.widgets import Button
import json

# Summary file
with open('summary_info.json', 'r') as file:
    summary_info = json.load(file)


class halo_data:
    cbar_lim = {'beta_raw': [-8, -4], 'v_raw': [-2, 2],
                'beta_averaged': [-8, -4], 'v_raw_averaged': [-2, 2],
                'cross_signal': [0.995, 1.005],
                'co_signal': [0.995, 1.005], 'depo_raw': [0, 0.5],
                'depo_averaged_raw': [0, 0.5],
                'depo_averaged': [0, 0.5],
                'co_signal_averaged': [0.995, 1.005],
                'cross_signal_averaged': [0.995, 1.005]}
    units = {'beta_raw': '$\\log (m^{-1} sr^{-1})$', 'v_raw': '$m s^{-1}$',
             'v_raw_averaged': '$m s^{-1}$',
             'beta_averaged': '$\\log (m^{-1} sr^{-1})$',
             'v_error': '$m s^{-1}$'}

    def __init__(self, path):
        self.full_data = netcdf.NetCDFFile(path, 'r', mmap=False)
        self.full_data_names = list(self.full_data.variables.keys())
        self.info = {name: self.full_data.variables[name].getValue(
        ) for name in self.full_data_names
            if self.full_data.variables[name].shape == ()}
        self.data = {name: self.full_data.variables[name].data
                     for name in self.full_data_names
                     if self.full_data.variables[name].shape != ()}
        self.data_names = list(self.data.keys())
        self.more_info = bytes_Odict_convert(self.full_data._attributes)

        name = [int(self.more_info.get(key)) if key != 'location' else
                self.more_info.get(key) for
                key in ['year', 'month', 'day', 'location', 'systemID']]
        self.date = '-'.join([str(ele).zfill(2) for ele in name[:-2]])
        self.location = '-'.join([str(ele).zfill(2) for ele in name[-2:]])
        self.filename = self.date + '-' + self.location
        delta_t = self.data['time'][1:] - self.data['time'][:-1]
        self.integration_time = (np.median(delta_t)/2)*3600

        if '32' in self.location:
            for period in summary_info['32']:
                if (period['start_date'] <= self.date) & \
                        (self.date <= period['end_date']):
                    self.snr_sd = period['snr_sd']
                    self.bleed_through_mean = period['bleed_through']['mean']
                    self.bleed_through_sd = period['bleed_through']['sd']
        else:
            id = str(int(self.more_info['systemID']))
            self.snr_sd = summary_info[id]['snr_sd']
            self.bleed_through_mean = summary_info[id]['bleed_through']['mean']
            self.bleed_through_sd = summary_info[id]['bleed_through']['sd']

    def meta_data(self):
        return {
            key1: {
                key: bytes_Odict_convert(
                    self.full_data.variables[key1].__dict__[key])
                for key in self.full_data.variables[key1].__dict__
                if key != 'data'}
            for key1 in self.full_data_names}

    def plot(self, variables=None, ncol=None, size=None):
        if ncol is None:
            ncol = 1
            nrow = len(variables)
        else:
            nrow = -(-len(variables)//ncol)  # Round up
        fig, ax = plt.subplots(nrow, ncol, figsize=size,
                               sharex=True, sharey=True)
        if nrow != 1 and ncol != 1:
            ax = ax.flatten()
        for i, var in enumerate(variables):
            if 'beta' in var:
                val = np.log10(self.data.get(var)).T
            else:
                val = self.data.get(var).T

            if 'average' in var:
                xvar = self.data.get('time_averaged')
            else:
                xvar = self.data.get('time')
            yvar = self.data.get('range')
            if self.cbar_lim.get(var) is None:
                vmin = None
                vmax = None
            else:
                vmin = self.cbar_lim.get(var)[0]
                vmax = self.cbar_lim.get(var)[1]
            if nrow == 1 and ncol == 1:
                axi = ax
            else:
                axi = ax[i]
            p = axi.pcolormesh(xvar, yvar, val, cmap='jet',
                               vmin=vmin,
                               vmax=vmax)
            axi.set_xlim([0, 24])
            axi.yaxis.set_major_formatter(m_km_ticks())
            axi.set_title(var)
            cbar = fig.colorbar(p, ax=axi, fraction=0.05)
            cbar.ax.set_ylabel(self.units.get(var, None), rotation=90)
            cbar.ax.yaxis.set_label_position('left')
        fig.suptitle(self.filename,
                     size=30,
                     weight='bold')
        lab_ax = fig.add_subplot(111, frameon=False)
        lab_ax.tick_params(labelcolor='none', top=False, bottom=False,
                           left=False, right=False)
        lab_ax.set_xlabel('Time (h)', weight='bold')
        lab_ax.set_ylabel('Height (km)', weight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig

    def filter(self, variables=None, ref=None, threshold=None):
        for var in variables:
            self.data[var] = np.where(
                self.data[ref] > threshold,
                self.data[var],
                float('nan'))

    def filter_attenuation(self, variables=None,
                           ref=None, threshold=None, buffer=2):
        mask = self.data[ref] > threshold
        mask_row = np.argwhere(mask.any(axis=1)).reshape(-1)
        mask_col = np.nanargmax(self.data[ref][mask_row, :] > threshold,
                                axis=1)
        for row, col in zip(mask_row, mask_col):
            for var in variables:
                self.data[var][row, col+buffer:] = np.nan

    def filter_height(self):
        '''
        Remove first three columns of the matrix due to calibration,
        ask Ville for more info
        '''
        for var in self.data_names:
            if 'time' not in var and 'range' not in var:
                self.data[var] = self.data[var][:, 3:]
            if var == 'range':
                self.data[var] = self.data[var][3:]

    def unmask999(self):
        '''
        Unmasked -999 values to nan for all variables except 'co_signal'
        and 'co_signal_averaged'
        '''
        for var in self.data_names:
            if 'co_signal' not in var:
                self.data[var][self.data[var] == -999] = float('nan')

    def interpolate_nan(self):
        '''
        Interpolate nan values
        '''
        for j in range(self.data['cross_signal'].shape[1]):
            mask_j = np.isnan(self.data['cross_signal'][:, j])
            self.data['cross_signal'][mask_j, j] = np.interp(
                self.data['time'][mask_j],
                self.data['time'][~mask_j],
                self.data['cross_signal'][~mask_j, j])

    def average(self, n):
        '''
        Average data
        '''
        self.data['co_signal_averaged'] = nan_average(
            self.data['co_signal'], n)
        self.data['cross_signal_averaged'] = nan_average(
            self.data['cross_signal'], n)
        self.data['depo_averaged'] = (self.data['cross_signal_averaged'] - 1) / \
            (self.data['co_signal_averaged'] - 1)
        self.data['v_averaged'] = nan_average(self.data['v_raw'], n)
        self.data['time_averaged'] = self.data['time'][::n]

    def moving_average(self, n=3):
        for avg, raw in zip(['co_signal_averaged', 'cross_signal_averaged',
                             'v_raw_averaged'],
                            ['co_signal', 'cross_signal', 'v_raw']):
            self.data[avg] = ma(self.data[raw], n)
        self.data['depo_adj_averaged'] = (self.data['cross_signal_averaged'] - 1) / \
            (self.data['co_signal_averaged'] - 1)
        self.data['time_averaged'] = self.data['time']

    def depo_cross_adj(self):
        bleed = self.bleed_through_mean
        sigma_bleed = self.bleed_through_sd
        sigma_co, sigma_cross = self.snr_sd, self.snr_sd

        self.data['cross_signal'] = (self.data['cross_signal'] - 1) - \
            bleed * (self.data['co_signal'] - 1) + 1

        self.data['cross_signal_sd'] = np.sqrt(
            sigma_cross**2 +
            ((bleed * (self.data['co_signal'] - 1))**2 *
             ((sigma_bleed/bleed)**2 +
              (sigma_co/(self.data['co_signal'] - 1))**2))
        )
        self.data['depo_adj'] = (self.data['cross_signal'] - 1) / \
            (self.data['co_signal'] - 1)

        self.data['depo_adj_sd'] = np.sqrt(
            (self.data['depo_adj'])**2 *
            (
                (self.data['cross_signal_sd']/(self.data['cross_signal'] - 1))**2 +
                (sigma_co/(self.data['co_signal']-1))**2
            ))

    def beta_averaged(self):
        '''
        Calculate beta from co_signal_averaged, which take into account focus
        '''
        c_light = 299792458
        hnu = 1.28e-19
        alpha = 0.01

        self.info['lens_diameter'] = self.info.get('lens_diameter', 0.06)
        self.info['wavelength'] = self.info.get('wavelength', 1.5e-6)
        self.info['beam_energy'] = self.info.get('beam_energy', 0.00001)

        het_area = np.pi * (0.7 * self.info['lens_diameter'] / 2) ** 2

        if self.info['focus'] < 0:
            nt = het_area / self.info['wavelength'] / self.data['range']
        else:
            nt = het_area / self.info['wavelength'] * (1 / self.info['focus'] -
                                                       1 / self.data['range'])

        effective_area = het_area / (1 + nt ** 2)
        T = 10 ** (-1 * alpha * self.data['range'] / 5000)
        het_cnr = 0.7 * 0.4 * 0.6 * effective_area * c_light * \
            self.info['beam_energy'] * T / (2 * self.data['range'] ** 2 *
                                            hnu * self.info['bandwidth'])

        pr2 = (self.data['co_signal_averaged'] - 1) / het_cnr.T
        self.data['beta_averaged'] = pr2

    @staticmethod
    def decision_tree(depo_thres, beta_thres, v_thres,
                      depo, beta, v):
        '''
        Decision tree
        '''
        mask_depo, mask_beta, mask_v = True, True, True
        mask_depo_0 = depo > depo_thres[0] if depo_thres[0] is not None else True
        mask_depo_1 = depo < depo_thres[1] if depo_thres[1] is not None else True
        if depo_thres[0] is not None or depo_thres[1] is not None:
            mask_depo = mask_depo_0 & mask_depo_1

        mask_beta_0 = beta > beta_thres[0] if beta_thres[0] is not None else True
        mask_beta_1 = beta < beta_thres[1] if beta_thres[1] is not None else True
        if beta_thres[0] is not None or beta_thres[1] is not None:
            mask_beta = mask_beta_0 & mask_beta_1

        mask_v_0 = v > v_thres[0] if v_thres[0] is not None else True
        mask_v_1 = v < v_thres[1] if v_thres[1] is not None else True
        if v_thres[0] is not None or v_thres[1] is not None:
            mask_v = mask_v_0 & mask_v_1

        mask = mask_depo & mask_beta & mask_v
        return mask

    def describe(self):
        pd.set_option('display.float_format', lambda x: '%.5g' % x)
        var_avg = {var: self.data[var].flatten().astype('f8')
                   for var in self.data if self.data[var].ndim != 1 and
                   'average' in var}
        varn = {var: self.data[var].flatten().astype('f8')
                for var in self.data if self.data[var].ndim != 1 and
                'average' not in var}

        combined_data = pd.DataFrame.from_dict(varn)
        describ = combined_data.describe(percentiles=[.25, .5, .75, .95])
        na = combined_data.isna().sum()
        summary = describ.append(na.rename('Missing values'))

        combined_data_avg = pd.DataFrame.from_dict(var_avg)
        describ_avg = combined_data_avg.describe(percentiles=[.25, .5,
                                                              .75, .95])
        na_avg = combined_data_avg.isna().sum()
        summary_avg = describ_avg.append(na_avg.rename('Missing values'))
        return summary.join(summary_avg)

    def snr_filter(self, multiplier=3, multiplier_avg=3):
        fig = plt.figure(figsize=(18, 9))
        spec = fig.add_gridspec(2, 2, width_ratios=[1, 1],
                                height_ratios=[2, 1])
        ax1 = fig.add_subplot(spec[0, 0])
        ax2 = fig.add_subplot(spec[0, 1])
        ax3 = fig.add_subplot(spec[1, 0])
        ax4 = fig.add_subplot(spec[1, 1], sharex=ax3)

        p1 = ax1.pcolormesh(self.data['time'],
                            self.data['range'],
                            self.data['co_signal'].T,
                            cmap='jet', vmin=0.995, vmax=1.005)
        ax1.yaxis.set_major_formatter(m_km_ticks())
        ax1.set_title('Choose background noise for co_signal')
        ax1.set_ylabel('Height (km)')
        ax1.set_xlabel('Time (h)')
        ax1.set_ylim(bottom=0)
        self.area_snr = area_snr(self.data['time'],
                                 self.data['range'],
                                 self.data['co_signal'].T,
                                 ax1,
                                 ax3,
                                 type='kde',
                                 multiplier=multiplier,
                                 fig=fig)
        fig.colorbar(p1, ax=ax1)

        p2 = ax2.pcolormesh(self.data['time_averaged'],
                            self.data['range'],
                            self.data['co_signal_averaged'].T,
                            cmap='jet', vmin=0.995, vmax=1.005)
        ax2.yaxis.set_major_formatter(m_km_ticks())
        ax2.set_title('Choose background noise for co_signal averaged')
        ax2.set_ylabel('Height (km)')
        ax2.set_xlabel('Time (h)')
        ax2.set_ylim(bottom=0)
        self.area_snr_avg = area_snr(
            self.data['time_averaged'],
            self.data['range'],
            self.data['co_signal_averaged'].T,
            ax2,
            ax4,
            type='kde',
            multiplier=multiplier_avg,
            fig=fig)
        fig.colorbar(p2, ax=ax2)
        fig.suptitle(self.filename, size=22,
                     weight='bold')

    def snr_save(self, snr_folder):
        with open(snr_folder + '/' + self.filename + '_noise.csv', 'w') as f:
            noise_area = self.area_snr.area.flatten()
            noise_shape = noise_area.shape
            noise_csv = pd.DataFrame.from_dict(
                {'year': np.repeat(self.more_info['year'], noise_shape),
                 'month': np.repeat(self.more_info['month'], noise_shape),
                 'day': np.repeat(self.more_info['day'], noise_shape),
                 'location': np.repeat(self.more_info['location'],
                                       noise_shape),
                 'systemID': np.repeat(self.more_info['systemID'],
                                       noise_shape),
                 'noise': noise_area - 1})
            noise_csv.to_csv(f, header=f.tell() == 0, index=False)

        with open(snr_folder + '/' + self.filename +
                  '_noise_avg' + '.csv', 'w') as ff:
            noise_area = self.area_snr_avg.area.flatten()
            noise_avg_shape = noise_area.shape
            noise_avg_csv = pd.DataFrame.from_dict(
                {'year': np.repeat(self.more_info['year'], noise_avg_shape),
                 'month': np.repeat(self.more_info['month'], noise_avg_shape),
                 'day': np.repeat(self.more_info['day'], noise_avg_shape),
                 'location': np.repeat(self.more_info['location'],
                                       noise_avg_shape),
                 'systemID': np.repeat(self.more_info['systemID'],
                                       noise_avg_shape),
                 'noise': noise_area - 1})
            noise_avg_csv.to_csv(ff, header=ff.tell() == 0, index=False)

    def depo_timeprofile(self):
        fig = plt.figure(figsize=(18, 9))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(223)
        ax3 = fig.add_subplot(224, sharey=ax2)
        p = ax1.pcolormesh(self.data['time'],
                           self.data['range'],
                           np.log10(self.data['beta_raw'].T),
                           cmap='jet', vmin=self.cbar_lim['beta_raw'][0],
                           vmax=self.cbar_lim['beta_raw'][1])
        fig.colorbar(p, ax=ax1, fraction=0.05, pad=0.02)
        ax1.set_title('beta_raw')
        ax1.set_xlabel('Time (h)')
        ax1.set_xlim([0, 24])
        ax1.set_ylabel('Height (km)')
        ax1.yaxis.set_major_formatter(m_km_ticks())
        fig.suptitle(self.filename,
                     size=30,
                     weight='bold')
        self.depo_tp = area_timeprofile(self.data['time'],
                                        self.data['range'],
                                        self.data['depo_raw'].T,
                                        ax1,
                                        ax_snr=ax3,
                                        ax_depo=ax2,
                                        snr=self.data['co_signal'].T,
                                        fig=fig)
        return fig

    def depo_timeprofile_save(self, fig, depo_folder):
        i = self.depo_tp.i
        area_value = self.depo_tp.area[:, i]
        area_range = self.data['range'][self.depo_tp.maskrange]
        area_snr = self.data['co_signal'].T[self.depo_tp.mask][:, i]
        area_vraw = self.data['v_raw'].T[self.depo_tp.mask][:, i]
        area_betaraw = self.data['beta_raw'].T[self.depo_tp.mask][:, i]
        area_cross = self.data['cross_signal'].T[self.depo_tp.mask][:, i]

        # Calculate indice of maximum snr value
        max_i = np.argmax(area_snr)

        result = pd.DataFrame.from_dict([{
            'year': self.more_info['year'],
            'month': self.more_info['month'],
            'day': self.more_info['day'],
            'location': self.more_info['location'],
            'systemID': self.more_info['systemID'],
            'time': self.data['time'][self.depo_tp.masktime][i],  # hour
            'range': area_range[max_i],  # range
            'depo': area_value[max_i],  # depo value
            'depo_1': area_value[max_i - 1],
            'co_signal': area_snr[max_i],  # snr
            'co_signal1': area_snr[max_i-1],
            'vraw': area_vraw[max_i],  # v_raw
            'beta_raw': area_betaraw[max_i],  # beta_raw
            'cross_signal': area_cross[max_i]  # cross_signal
        }])

        # sub folder for each date
        depo_sub_folder = depo_folder + '/' + self.filename
        Path(depo_sub_folder).mkdir(parents=True, exist_ok=True)

        # Append to or create new csv file
        with open(depo_sub_folder + '/' +
                  self.filename + '_depo.csv', 'a') as f:
            result.to_csv(f, header=f.tell() == 0, index=False)
        # save fig
        fig.savefig(
            depo_sub_folder + '/' + self.filename + '_' +
            str(int(self.data['time'][self.depo_tp.masktime][i]*1000)) +
            '.png')

    def depo_wholeprofile(self):
        fig = plt.figure(figsize=(18, 9))
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(323)
        ax3 = fig.add_subplot(325, sharex=ax2)
        ax4 = fig.add_subplot(324)
        ax5 = fig.add_subplot(326)
        p = ax1.pcolormesh(self.data['time'],
                           self.data['range'],
                           np.log10(self.data['beta_raw'].T),
                           cmap='jet', vmin=self.cbar_lim['beta_raw'][0],
                           vmax=self.cbar_lim['beta_raw'][1])
        fig.colorbar(p, ax=ax1, fraction=0.05, pad=0.02)
        ax1.set_title('beta_raw')
        ax1.set_xlabel('Time (h)')
        ax1.set_xlim([0, 24])
        ax1.set_ylabel('Height (km)')
        ax1.yaxis.set_major_formatter(m_km_ticks())
        fig.suptitle(self.filename,
                     size=30,
                     weight='bold')
        fig.subplots_adjust(hspace=0.3)
        self.depo_wp = area_wholecloud(self.data['time'],
                                       self.data['range'],
                                       self.data['depo_raw'].T,
                                       ax1,
                                       ax_snr=ax3,
                                       ax_depo=ax2,
                                       ax_hist_depo=ax4,
                                       ax_hist_snr=ax5,
                                       snr=self.data['co_signal'].T,
                                       fig=fig)
        return fig

    def depo_wholeprofile_save(self, fig, depo_folder):
        n_values = self.depo_wp.time.shape[0]
        result = pd.DataFrame.from_dict({
            'year': np.repeat(self.more_info['year'], n_values),
            'month': np.repeat(self.more_info['month'], n_values),
            'day': np.repeat(self.more_info['day'], n_values),
            'location': np.repeat(self.more_info['location'], n_values),
            'systemID': np.repeat(self.more_info['systemID'], n_values),
            'time': self.depo_wp.time,  # time as hour
            'range': self.depo_wp.range[self.depo_wp.max_snr_indx][0],  # range
            'depo': self.depo_wp.depo_max_snr,  # depo value
            'depo_1': self.depo_wp.depo_max_snr1,
            'co_signal': self.depo_wp.max_snr,  # snr
            'co_signal1': self.depo_wp.max_snr1,
            'vraw': np.take_along_axis(self.data['v_raw'].T[self.depo_wp.mask],
                                       self.depo_wp.max_snr_indx,
                                       axis=0)[0],  # v_raw
            'beta_raw': np.take_along_axis(
                self.data['beta_raw'].T[self.depo_wp.mask],
                self.depo_wp.max_snr_indx,
                axis=0)[0],  # beta_raw
            'cross_signal': np.take_along_axis(
                self.data['cross_signal'].T[self.depo_wp.mask],
                self.depo_wp.max_snr_indx,
                axis=0)[0]  # cross_signal
        })

        # sub folder for each date
        depo_sub_folder = depo_folder + '/' + self.filename
        Path(depo_sub_folder).mkdir(parents=True, exist_ok=True)

        # Append to or create new csv file
        with open(depo_sub_folder + '/' +
                  self.filename + '_depo.csv', 'a') as f:
            result.to_csv(f, header=f.tell() == 0, index=False)
        # save fig
        fig.savefig(depo_sub_folder + '/' + self.filename + '_' +
                    f'{self.depo_wp.time.min()*100:.0f}' + '-' +
                    f'{self.depo_wp.time.max()*100:.0f}' + '.png')

    def v_along_time(self, size=51, height=2000, v='v_raw'):
        height_idx = np.argmin(self.data['range'] < height)
        v_ = self.data[v][:, :height_idx]
        shape = v_.shape
        z = np.full([shape[0]+size, shape[1] * size], np.nan)
        for i in range(size):
            z[i:(shape[0]+i),
              (shape[1] * i):(shape[1] * (i + 1))] = v_

        bound = int(np.floor(size/2))
        self.v_std = np.nanstd(z, axis=1)
        self.v_std = self.v_std[bound+1:-bound]
        self.v_mean = np.nanmean(z, axis=1)
        self.v_mean = self.v_mean[bound+1:-bound]

    def cross_signal_sd(self):
        fig, ax = plt.subplots(1, 2, figsize=(18, 9))
        p = ax[0].pcolormesh(self.data['time'], self.data['range'],
                             self.data['cross_signal'].T, cmap='jet',
                             vmin=0.995, vmax=1.005)
        fig.colorbar(p, ax=ax[0])
        ax[0].yaxis.set_major_formatter(m_km_ticks())
        ax[0].set_title('Choose background noise for cross_signal')
        ax[0].set_ylabel('Height (km)')
        ax[0].set_xlabel('Time (h)')
        ax[0].set_ylim(bottom=0)
        self.area_cross_signal = area_snr(
            self.data['time'],
            self.data['range'],
            self.data['cross_signal'].T,
            ax[0],
            ax[1],
            type='kde',
            multiplier=3,
            fig=fig)
        fig.suptitle(self.filename, size=22,
                     weight='bold')

    def cross_signal_sd_save(self, cross_signal_folder):
        with open(cross_signal_folder + '/' + self.filename +
                  '_cross_signal.csv', 'w') as f:
            noise_area = self.area_cross_signal.area.flatten()
            noise_shape = noise_area.shape
            noise_csv = pd.DataFrame.from_dict(
                {'year': np.repeat(self.more_info['year'], noise_shape),
                 'month': np.repeat(self.more_info['month'], noise_shape),
                 'day': np.repeat(self.more_info['day'], noise_shape),
                 'location': np.repeat(self.more_info['location'],
                                       noise_shape),
                 'systemID': np.repeat(self.more_info['systemID'],
                                       noise_shape),
                 'noise': noise_area - 1})
            noise_csv.to_csv(f, header=f.tell() == 0, index=False)


class area_select():

    def __init__(self, x, y, z, ax_in, fig):
        self.x, self.y, self.z = x, y, z
        self.ax_in = ax_in
        self.canvas = fig.canvas
        self.fig = fig
        self.selector = RectangleSelector(
            self.ax_in,
            self,
            useblit=True,  # Process much faster,
            interactive=True  # Keep the drawn box on screen
        )

    def __call__(self, event1, event2):
        self.mask = self.inside(event1, event2)
        self.area = self.z[self.mask]
        self.range = self.y[self.maskrange]
        self.time = self.x[self.masktime]
        print(f'Chosen {len(self.area.flatten())} values')

    def inside(self, event1, event2):
        """
        Returns a boolean mask of the points inside the rectangle defined by
        event1 and event2
        """
        self.xcord = [event1.xdata, event2.xdata]
        self.ycord = [event1.ydata, event2.ydata]
        x0, x1 = sorted(self.xcord)
        y0, y1 = sorted(self.ycord)
        self.masktime = (self.x > x0) & (self.x < x1)  # remove bracket ()
        self.maskrange = (self.y > y0) & (self.y < y1)
        return np.ix_(self.maskrange, self.masktime)


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


class span_depo_select(span_select):

    def __init__(self, x, y, ax_in, canvas, orient, depo_sd):
        super().__init__(x, y, ax_in, canvas, orient)
        self.depo_sd = depo_sd

    def __call__(self, min, max):
        super().__call__(min, max)
        print(f're-selected depo: {np.nanmean(self.selected_x):.3f}')
        print(f're-selected depo std: {np.nanmean(self.depo_sd[self.masky]):.3f}')


class area_aerosol(area_select):

    def __init__(self, x, y, z, ax_in, fig, ax2, df, ax3, ax4):
        super().__init__(x, y, z, ax_in, fig)
        self.df = df
        self.co = self.df.data['co_signal']
        self.cross = self.df.data['cross_signal']
        self.ax2, self.ax3, self.ax4 = ax2, ax3, ax4

    def __call__(self, event1, event2):
        super().__call__(event1, event2)
        self.ax2.cla()
        self.ax3.cla()
        self.ax4.cla()
        self.t = (self.x >= self.xcord[0]) & (self.x <= self.xcord[1])
        self.range_aerosol = (self.y > self.ycord[0]) & (self.y < self.ycord[1])
        self.co_mean_profile = np.nanmean(self.co[self.t, :], axis=0)
        self.cross_mean_profile = np.nanmean(self.cross[self.t, :], axis=0)
        self.ax2.plot(self.co_mean_profile,
                      self.y, '.', label='co_signal', c='red')
        self.ax2.plot(self.cross_mean_profile,
                      self.y, '.', label='cross_signal', c='blue')
        self.ax2.yaxis.set_major_formatter(m_km_ticks())
        self.ax2.legend()
        self.ax2.set_xlim([0.9995, 1.003])
        self.ax2.set_title('Co and cross profile')
        self.ax2.set_ylabel('Height (km)')
        self.ax2.set_xlabel('SNR')
        self.axapply = self.fig.add_axes([0.7, 0.005, 0.1, 0.025])
        self.axfit = self.fig.add_axes([0.81, 0.005, 0.1, 0.025])
        self.bfit = Button(self.axfit, 'Fit')
        self.bapply = Button(self.axapply, 'Apply')
        self.canvas.draw()
        self.span_aerosol = span_aerosol(self.co_mean_profile, self.y, self.ax2,
                                         self.canvas, 'vertical',
                                         self.cross_mean_profile,
                                         self.range_aerosol,
                                         self.df, self.ax3, self.ax4)
        self.bfit.on_clicked(self.span_aerosol.fit)
        self.bapply.on_clicked(self.span_aerosol.apply)


class span_aerosol(span_select):

    def __init__(self, x, y, ax_in, canvas, orient, cross, range_aerosol, df,
                 ax3, ax4):
        super().__init__(x, y, ax_in, canvas, orient)
        self.range_aerosol, self.df = range_aerosol, df
        self.co, self.range, self.cross = x, y, cross
        self.co_all = np.array([])
        self.cross_all = np.array([])
        self.range_all = np.array([])
        self.ax3, self.ax4 = ax3, ax4

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

        self.ax_in.plot(y_co, self.range, c='red')
        self.ax_in.plot(y_cross, self.range, c='blue')

        co_corrected = self.co/y_co
        self.co_corrected = co_corrected
        cross_corrected = self.cross/y_cross
        cross_sd_background = np.nanstd(cross/y_cross_background)
        co_sd_background = np.nanstd(co/y_co_background)
        self.sigma_co, self.sigma_cross = co_sd_background, cross_sd_background
        bleed = self.df.bleed_through_mean
        sigma_bleed = self.df.bleed_through_sd

        self.cross_corrected = (cross_corrected - 1) - \
            self.df.bleed_through_mean * (co_corrected - 1) + 1

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

        self.depo_corrected[co_corrected < 1 + 3*co_sd_background] = np.nan
        self.depo_corrected_sd[co_corrected < 1 + 3*co_sd_background] = np.nan
        self.final_depo = np.nanmean(self.depo_corrected[self.range_aerosol])
        self.final_depo_sd = np.nanmean(self.depo_corrected_sd[self.range_aerosol])
        print(f'Depo: {self.final_depo:.3f} with std: {self.final_depo_sd:.3f}')
        self.ax3.plot(self.co_corrected,
                      self.range, '.', label='co_signal', c='red')
        self.ax3.plot(self.cross_corrected,
                      self.range, '.', label='cross_signal', c='blue')
        self.ax3.set_xlim([0.9995, 1.003])
        self.ax3.legend()
        self.ax3.set_title('Corrected co and cross')
        self.ax3.set_xlabel('SNR')
        self.ax4.errorbar(self.depo_corrected,
                          self.range, xerr=self.depo_corrected_sd,
                          errorevery=1, elinewidth=0.5, fmt='.')
        self.ax4.set_xlabel('Depolarization ratio')
        self.ax4.set_title('Averaged depo profile')
        self.ax4.set_xlim([-0.1, 0.5])
        self.canvas.draw()
        self.depo_select = span_depo_select(self.depo_corrected, self.range,
                                            self.ax4, self.canvas, 'vertical',
                                            self.depo_corrected_sd)


class area_snr(area_select):

    def __init__(self, x, y, z, ax_in, ax_snr, fig, type='hist', multiplier=3):
        super().__init__(x, y, z, ax_in, fig)
        self.ax_snr = ax_snr
        self.type = type
        self.multiplier = multiplier

    def __call__(self, event1, event2):
        super().__call__(event1, event2)
        self.ax_snr.cla()
        if self.type == 'hist':
            self.ax_snr.hist(self.area.flatten())
        elif self.type == 'kde':
            sns.kdeplot(self.area.flatten(), ax=self.ax_snr)
        area_mean = np.nanmean(self.area)
        self.threshold = 1 + np.nanstd(self.area - 1) * self.multiplier
        self.ax_snr.set_xlabel(
            f'mean: {area_mean:.7f} \n threshold: {self.threshold:.7f}')
        print('Calculated threshold is: ', self.threshold)
        self.ax_snr.axvline(area_mean, c='red')
        self.canvas.draw()


class area_timeprofile(area_select):

    def __init__(self, x, y, z, ax_in, ax_snr, ax_depo, snr, fig,
                 ax_snr_label='SNR', ax_depo_label='Depo'):
        super().__init__(x, y, z, ax_in, fig)
        self.ax_depo = ax_depo
        self.ax_snr = ax_snr
        self.i = -1
        self.snr = snr
        self.ax_snr_label = ax_snr_label
        self.ax_depo_label = ax_depo_label

    def __call__(self, event1, event2):
        super().__call__(event1, event2)
        self.canvas.mpl_connect('key_press_event', self.key_interact)

    def key_interact(self, event):
        self.ax_depo.cla()
        self.ax_snr.cla()
        self.i += 1
        snr_profile = self.snr[self.mask][:, self.i]
        time_point = self.time[self.i]
        self.ax_depo.scatter(self.area[:, self.i],
                             self.range,
                             c=snr_profile,
                             s=snr_profile*20)
        self.ax_snr.scatter(snr_profile,
                            self.range,
                            c=snr_profile,
                            s=snr_profile*20)
        self.ax_depo.set_xlabel(
            f"Depo value colored by SNR at time {time_point:.3f}")
        self.ax_snr.set_xlabel(f"SNR at time {time_point:.3f}")
        self.ax_depo.set_ylabel('Height (m)')
        self.canvas.draw()


class area_wholecloud(area_select):

    def __init__(self, x, y, z, ax_in,
                 ax_depo, ax_snr, fig,
                 ax_hist_depo, ax_hist_snr,
                 snr):
        super().__init__(x, y, z, ax_in, fig)
        self.ax_depo = ax_depo
        self.ax_snr = ax_snr
        self.snr = snr
        self.ax_hist_depo = ax_hist_depo
        self.ax_hist_snr = ax_hist_snr

    def __call__(self, event1, event2):
        super().__call__(event1, event2)
        self.ax_depo.cla()
        self.ax_snr.cla()
        self.ax_hist_depo.cla()
        self.ax_hist_snr.cla()

        self.area_snr = self.snr[self.mask]
        self.max_snr_indx = np.expand_dims(np.nanargmax(self.area_snr, axis=0),
                                           axis=0)
        self.max_snr = np.take_along_axis(self.area_snr,
                                          self.max_snr_indx,
                                          axis=0)[0]
        self.max_snr1 = np.take_along_axis(self.area_snr,
                                           self.max_snr_indx - 1,
                                           axis=0)[0]
        self.depo_max_snr = np.take_along_axis(self.area,
                                               self.max_snr_indx,
                                               axis=0)[0]
        self.depo_max_snr1 = np.take_along_axis(self.area,
                                                self.max_snr_indx - 1,
                                                axis=0)[0]

        self.ax_depo.plot(self.time, self.depo_max_snr, '-',
                          label='depo at maxSNR')
        self.ax_depo.plot(self.time, self.depo_max_snr1, '--',
                          alpha=0.3, label='depo at maxSNR-1')
        self.ax_depo.set_ylabel('Depo value')
        self.ax_depo.set_title('Depo time series in selected area',
                               weight='bold')
        self.ax_depo.legend()

        self.ax_snr.plot(self.time, self.max_snr, '-', label='maxSNR')
        self.ax_snr.plot(self.time, self.max_snr1, '--', label='maxSNR-1')
        self.ax_snr.legend()
        self.ax_snr.set_xlabel('Time (h)')
        self.ax_snr.set_ylabel('SNR')
        self.ax_snr.set_title('SNR time series in selected area',
                              weight='bold')

        sns.kdeplot(self.max_snr, label='maxSNR', ax=self.ax_hist_snr)
        sns.kdeplot(self.max_snr1, label='maxSNR-1',
                    linestyle="--", ax=self.ax_hist_snr)
        self.ax_hist_snr.set_title('SNR distribution', weight='bold')

        sns.kdeplot(self.depo_max_snr, label='depo at maxSNR',
                    ax=self.ax_hist_depo)
        sns.kdeplot(self.depo_max_snr1, label='depo at maxSNR-1',
                    linestyle="--", ax=self.ax_hist_depo)
        self.ax_hist_depo.set_title('Depo distribution', weight='bold')

        self.canvas.draw()


def getdata(path):
    '''
    Get .nc files from a path
    '''
    import glob
    data_list = glob.glob(path + "/*.nc")
    return sorted(data_list)


def getdata_date(data, date, data_folder):
    data_indices = [i for i, s in enumerate(data) if date
                    in s.replace(data_folder, '')]
    assert len(data_indices) == 1, 'There are more than 1 data have that date'
    return data_indices[0]


def m_km_ticks():
    '''
    Modify ticks from m to km
    '''
    return FuncFormatter(lambda x, pos: f'{x/1000:.0f}')


def bytes_Odict_convert(x):
    '''
    Convert order dict to dict, and bytes to normal string
    '''
    if type(x) is OrderedDict:
        x = dict(x)
        x = {key: bytes_Odict_convert(val) for key, val in x.items()}
    if type(x) == bytes:
        return x.decode('utf-8')
    else:
        return x


def ma(a, n=3, axis=0):
    '''
    Moving average
    '''
    pad = int((n-1)/2)
    sum_ = np.nancumsum(a, axis=0)
    count_ = np.nancumsum(~np.isnan(a), axis=0)
    b = a.copy()
    b[pad+1:-pad] = (sum_[n:, :] - sum_[:-n, :]) / \
        (count_[n:, :] - count_[:-n, :])
    b[pad] = sum_[n - 1]/count_[n - 1]
    sum_0 = sum_[pad:n-1, :]
    sum_1 = sum_[-1, :] - sum_[-n:-pad-1, :]
    count_0 = count_[pad:n-1, :]
    count_1 = count_[-1, :] - count_[-n:-pad-1, :]
    b[:pad] = sum_0/count_0
    b[-pad:] = sum_1/count_1
    return b


def nan_average(data, n):
    x, y = data.shape
    temp = np.full((n*int(np.ceil(x/n)), y), np.nan)
    temp[:x, :y] = data
    return np.nanmean(temp.reshape(-1, n, y), axis=1)


def aggregate_data(nc_path,
                   start_date, end_date,
                   cloud_thres=10**-4.5, cloud_buffer=2,
                   attenuation=True):
    '''
    Aggerate and preprocess data
    date format: '%Y-%m-%d'
    '''
    import glob
    date_range = pd.date_range(start=start_date,
                               end=end_date).strftime('%Y%m%d')

    data_list = glob.glob(nc_path + '/*.nc')

    depo_list = []
    v_list = []
    beta_list = []
    date_list = []
    range_list = {}
    time_list = {}

    for date in date_range:
        file = [file for file in data_list if date in file]
        if len(file) == 0:
            print(f'{date} is missing')
            continue
        elif len(file) > 1:
            print(f'There are two {date}')
            break

        file = file[0]

        df = halo_data(file)
        df.unmask999()
        df.filter_height()
        df.depo_cross_adj()
        df.average(25)
        df.beta_averaged()
        noise_threshold = 1 + 3 * df.snr_sd/5
        df.filter(variables=['beta_averaged', 'v_averaged', 'depo_averaged',
                             'cross_signal_averaged'],
                  ref='co_signal_averaged', threshold=noise_threshold)

        if attenuation:
            df.filter_attenuation(variables=['beta_raw', 'v_raw', 'depo_raw',
                                             'cross_signal_averaged'],
                                  ref='beta_averaged',
                                  threshold=cloud_thres, buffer=cloud_buffer)

        for depo_value in df.data['depo_averaged'].ravel():
            depo_list.append(depo_value)
        for v_value in df.data['v_averaged'].ravel():
            v_list.append(v_value)
        b = df.data['beta_averaged'].ravel()
        for beta_value in b:
            beta_list.append(beta_value)
        for date_value in np.repeat(date, len(b)):
            date_list.append(date_value)

        range_list[date] = df.data['range']
        time_list[date] = df.data['time_averaged']

    result = pd.DataFrame({'depo': depo_list, 'v_raw': v_list,
                           'beta_raw': beta_list, 'date': date_list})
    return result, time_list, range_list


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


# %% Wavelet background method


def background_detection(x, threshold, wavelet='bior2.6'):
    coeff = pywt.swt(np.pad(x-1, (0, (len(x) // 2**5 + 3) * 2**5 - len(x)), 'constant', constant_values=(0, 0)),
                     wavelet, level=5, trim_approx=True)
    uthresh = np.median(np.abs(coeff[1]))/0.6745 * np.sqrt(2 * np.log(len(coeff[1])))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    filtered = pywt.iswt(coeff, wavelet) + 1
    filtered = filtered[:len(x)]
    filtered_ = np.convolve(filtered, [1/5, 1/5, 1/5, 1/5, 1/5], 'same')
    filtered_[:3] = filtered[:3]
    filtered_[-3:] = filtered[-3:]
    filtered = filtered_
    # peaks, _ = find_peaks(-filtered)
    # peaks = peaks[filtered[peaks] < 1]
    #
    # indices = np.arange(len(x))
    # a, b, c = np.polyfit(peaks, filtered[peaks], deg=2)
    # smooth_peaks = c + b*indices + a*(indices**2)
    # background = filtered < smooth_peaks + 6e-5*2
    background = filtered < 1 + threshold

    # return background, peaks, smooth_peaks
    return background


def background_correction(x, background):
    indices = np.arange(len(x))
    selected_x = x[background]
    selected_indices = indices[background]

    a, b, c = np.polyfit(selected_indices, selected_x, deg=2)
    x_fitted = c + b*indices + a*(indices**2)
    x_background_fitted = c + b*selected_indices + a*(selected_indices**2)

    x_corrected = x/x_fitted
    x_background_corrected = selected_x/x_background_fitted

    return x_corrected, x_background_corrected, x_fitted
