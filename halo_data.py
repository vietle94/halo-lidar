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


class halo_data:
    cbar_lim = {'beta_raw': [-8, -4], 'v_raw': [-2, 2],
                'cross_signal': [0.995, 1.005],
                'co_signal': [0.995, 1.005], 'depo_raw': [0, 0.5],
                'depo_averaged_raw': [0, 0.5], 'co_signal_averaged': [0.995, 1.005],
                'cross_signal_averaged': [0.995, 1.005]}
    units = {'beta_raw': '$\log (m^{-1} sr^{-1})$', 'v_raw': '$m s^{-1}$',
             'v_error': '$m s^{-1}$'}

    def __init__(self, path):
        self.full_data = netcdf.NetCDFFile(path, 'r', mmap=False)
        self.full_data_names = list(self.full_data.variables.keys())
        self.info = {name: self.full_data.variables[name].getValue(
        ) for name in self.full_data_names if self.full_data.variables[name].shape == ()}
        self.data = {name: self.full_data.variables[name].data
                     for name in self.full_data_names if self.full_data.variables[name].shape != ()}
        self.data_names = list(self.data.keys())
        self.more_info = bytes_Odict_convert(self.full_data._attributes)

        name = [int(self.more_info.get(key)) if key != 'location' else
                self.more_info.get(key) for
                key in ['year', 'month', 'day', 'location', 'systemID']]

        self.filename = '-'.join([str(elem).zfill(2) if name in ['month',
                                                                 'day'] else str(elem).zfill(2) for elem in name])
        self.meta_data = {key1: {key: bytes_Odict_convert(self.full_data.variables[key1].__dict__[key])
                                 for key in self.full_data.variables[key1].__dict__ if key != 'data'}
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
            if 'beta_raw' in var:
                val = np.log10(self.data.get(var)).transpose()
            else:
                val = self.data.get(var).transpose()

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
        lab_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
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

    def describe(self):
        import pandas as pd
        pd.set_option('display.float_format', lambda x: '%.5g' % x)
        var_avg = {var: self.data[var].flatten().astype('f8')
                   for var in self.data if self.data[var].ndim != 1 and 'average' in var}
        varn = {var: self.data[var].flatten().astype('f8')
                for var in self.data if self.data[var].ndim != 1 and 'average' not in var}

        combined_data = pd.DataFrame.from_dict(varn)
        describ = combined_data.describe(percentiles=[.25, .5, .75, .95])
        na = combined_data.isna().sum()
        summary = describ.append(na.rename('Missing values'))

        combined_data_avg = pd.DataFrame.from_dict(var_avg)
        describ_avg = combined_data_avg.describe(percentiles=[.25, .5, .75, .95])
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
                            self.data['co_signal'].transpose(),
                            cmap='jet', vmin=0.995, vmax=1.005)
        ax1.yaxis.set_major_formatter(m_km_ticks())
        ax1.set_title('Choose background noise for co_signal')
        ax1.set_ylabel('Height (km)')
        ax1.set_xlabel('Time (h)')
        ax1.set_ylim(bottom=0)
        self.area_snr = area_snr(self.data['time'],
                                 self.data['range'],
                                 self.data['co_signal'].transpose(),
                                 ax1,
                                 ax3,
                                 type='kde',
                                 multiplier=multiplier,
                                 fig=fig)
        fig.colorbar(p1, ax=ax1)

        p2 = ax2.pcolormesh(self.data['time_averaged'],
                            self.data['range'],
                            self.data['co_signal_averaged'].transpose(),
                            cmap='jet', vmin=0.995, vmax=1.005)
        ax2.yaxis.set_major_formatter(m_km_ticks())
        ax2.set_title('Choose background noise for co_signal averaged')
        ax2.set_ylabel('Height (km)')
        ax2.set_xlabel('Time (h)')
        ax2.set_ylim(bottom=0)
        self.area_snr_avg = area_snr(self.data['time_averaged'],
                                     self.data['range'],
                                     self.data['co_signal_averaged'].transpose(),
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
            noise_csv = pd.DataFrame.from_dict({'year': np.repeat(self.more_info['year'], noise_shape),
                                                'month': np.repeat(self.more_info['month'], noise_shape),
                                                'day': np.repeat(self.more_info['day'], noise_shape),
                                                'location': np.repeat(self.more_info['location'], noise_shape),
                                                'systemID': np.repeat(self.more_info['systemID'], noise_shape),
                                                'noise': noise_area - 1})
            noise_csv.to_csv(f, header=f.tell() == 0, index=False)

        with open(snr_folder + '/' + self.filename + '_noise_avg' + '.csv', 'w') as ff:
            noise_area = self.area_snr_avg.area.flatten()
            noise_avg_shape = noise_area.shape
            noise_avg_csv = pd.DataFrame.from_dict({'year': np.repeat(self.more_info['year'], noise_avg_shape),
                                                    'month': np.repeat(self.more_info['month'], noise_avg_shape),
                                                    'day': np.repeat(self.more_info['day'], noise_avg_shape),
                                                    'location': np.repeat(self.more_info['location'], noise_avg_shape),
                                                    'systemID': np.repeat(self.more_info['systemID'], noise_avg_shape),
                                                    'noise': noise_area - 1})
            noise_avg_csv.to_csv(ff, header=ff.tell() == 0, index=False)

    def depo_timeprofile(self):
        fig = plt.figure(figsize=(18, 9))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(223)
        ax3 = fig.add_subplot(224, sharey=ax2)
        p = ax1.pcolormesh(self.data['time'],
                           self.data['range'],
                           np.log10(self.data['beta_raw'].transpose()),
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
                                        self.data['depo_raw'].transpose(),
                                        ax1,
                                        ax_snr=ax3,
                                        ax_depo=ax2,
                                        snr=self.data['co_signal'].transpose(),
                                        fig=fig)
        return fig

    def depo_timeprofile_save(self, fig, depo_folder):
        i = self.depo_tp.i
        area_value = self.depo_tp.area[:, i]
        area_range = self.data['range'][self.depo_tp.maskrange]
        area_snr = self.data['co_signal'].transpose()[self.depo_tp.mask][:, i]
        area_vraw = self.data['v_raw'].transpose()[self.depo_tp.mask][:, i]
        area_betaraw = self.data['beta_raw'].transpose()[self.depo_tp.mask][:, i]
        area_cross = self.data['cross_signal'].transpose()[self.depo_tp.mask][:, i]

        # Calculate indice of maximum snr value
        max_i = np.argmax(area_snr)

        result = pd.DataFrame.from_dict([{
            'year': self.more_info['year'],
            'month': self.more_info['month'],
            'day': self.more_info['day'],
            'location': self.more_info['location'],
            'systemID': self.more_info['systemID'],
            'time': self.data['time'][self.depo_tp.masktime][i],  # time as hour
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
        with open(depo_sub_folder + '/' + self.filename + '_depo.csv', 'a') as f:
            result.to_csv(f, header=f.tell() == 0, index=False)
        # save fig
        fig.savefig(depo_sub_folder + '/' + self.filename + '_' +
                    str(int(self.data['time'][self.depo_tp.masktime][i]*1000)) + '.png')

    def depo_wholeprofile(self):
        fig = plt.figure(figsize=(18, 9))
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(323)
        ax3 = fig.add_subplot(325, sharex=ax2)
        ax4 = fig.add_subplot(324)
        ax5 = fig.add_subplot(326)
        p = ax1.pcolormesh(self.data['time'],
                           self.data['range'],
                           np.log10(self.data['beta_raw'].transpose()),
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
                                       self.data['depo_raw'].transpose(),
                                       ax1,
                                       ax_snr=ax3,
                                       ax_depo=ax2,
                                       ax_hist_depo=ax4,
                                       ax_hist_snr=ax5,
                                       snr=self.data['co_signal'].transpose(),
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
            'vraw': np.take_along_axis(self.data['v_raw'].transpose()[self.depo_wp.mask],
                                       self.depo_wp.max_snr_indx,
                                       axis=0)[0],  # v_raw
            'beta_raw': np.take_along_axis(self.data['beta_raw'].transpose()[self.depo_wp.mask],
                                           self.depo_wp.max_snr_indx,
                                           axis=0)[0],  # beta_raw
            'cross_signal': np.take_along_axis(self.data['cross_signal'].transpose()[self.depo_wp.mask],
                                               self.depo_wp.max_snr_indx,
                                               axis=0)[0]  # cross_signal
        })

        # sub folder for each date
        depo_sub_folder = depo_folder + '/' + self.filename
        Path(depo_sub_folder).mkdir(parents=True, exist_ok=True)

        # Append to or create new csv file
        with open(depo_sub_folder + '/' + self.filename + '_depo.csv', 'a') as f:
            result.to_csv(f, header=f.tell() == 0, index=False)
        # save fig
        fig.savefig(depo_sub_folder + '/' + self.filename + '_' +
                    f'{self.depo_wp.time.min()*100:.0f}' + '-' + f'{self.depo_wp.time.max()*100:.0f}' + '.png')

    # def depo_aerosol(self):
    #     fig, ax = plt.subplots(5, 1, figsize=(18, 9))
    #     p = ax[0].pcolormesh(self.data['time'],
    #                          self.data['range'],
    #                          np.log10(self.data['beta_raw'].transpose()),
    #                          cmap='jet', vmin=self.cbar_lim['beta_raw'][0],
    #                          vmax=self.cbar_lim['beta_raw'][1])
    #     fig.colorbar(p, ax=ax[0], fraction=0.05, pad=0.02)
    #     ax[0].set_title('beta_raw')
    #     ax[0].set_xlabel('Time (h)')
    #     ax[0].set_xlim([0, 24])
    #     ax[0].set_ylabel('Height (km)')
    #     ax[0].yaxis.set_major_formatter(m_km_ticks())
    #     fig.suptitle(self.filename,
    #                  size=30,
    #                  weight='bold')
    #     fig.subplots_adjust(hspace=0.3)
    #     self.aerosol = area_select


class area_select():

    def __init__(self, x, y, z, ax_in, fig):
        self.x, self.y, self.z = x, y, z
        self.ax_in = ax_in
        self.canvas = fig.canvas
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

    def __init__(self, x, y, ax_in, canvas, velocity, beta):
        self.x, self.y = x, y
        self.v, self.b = velocity, beta
        self.ax_in = ax_in
        self.canvas = canvas
        self.selector = SpanSelector(
            self.ax_in, self, 'horizontal', span_stays=True, useblit=True
        )

    def __call__(self, min, max):
        self.maskx = (self.x > min) & (self.x < max)
        # self.selected_x = self.x[self.maskx]
        self.selected_y = self.y[self.maskx]
        self.selected_v = self.v[self.maskx]
        self.selected_b = self.b[self.maskx]
        # self.not_selected_y = self.y[np.invert(self.maskx)]


class span_aerosol(span_select):

    def __init__(self, x, y, ax_in, canvas,
                 ax10, ax12, velocity, beta):
        super().__init__(x, y, ax_in, canvas, velocity, beta)
        self.ax10 = ax10
        self.ax12 = ax12

    def __call__(self, min, max):
        super().__call__(min, max)
        for ax in [self.ax10, self.ax12]:
            ax.cla()

        self.df = pd.DataFrame({'depo': self.selected_y,
                                'velocity': self.selected_v,
                                'beta': self.selected_b})
        self.df.dropna(inplace=True)
        self.df.beta = np.log10(self.df.beta)
        H, x_edges, y_edges = np.histogram2d(self.df['depo'],
                                             self.df['velocity'],
                                             bins=20)
        X, Y = np.meshgrid(x_edges, y_edges)
        p = self.ax10.pcolormesh(X, Y, H.T, norm=LogNorm())
        self.ax10.set_xlabel('depo')
        self.ax10.set_ylabel('velocity')

        # self.ax10.hist(self.selected_y.flatten())
        # self.ax10.set_title('Depo of chosen area')
        # self.ax10.set_xlabel('Depo')

        self.ax12.scatter(self.df['depo'], self.df['velocity'],
                          self.df['beta'])
        self.ax12.set_xlabel('Depo')
        self.ax12.set_ylabel('Velocity')
        self.ax12.set_zlabel('Beta')
        self.canvas.fig.colorbar(p, ax=self.ax10)

        # self.ax12.hist(self.not_selected_y.flatten())
        # self.ax12.set_title('Depo of unchosen area')
        # self.ax12.set_xlabel('Depo')
        self.canvas.draw()


class area_aerosol(area_select):

    def __init__(self, x, y, z, ax_in, fig,
                 snr, ax_out,
                 ax10, ax12, threshold,
                 velocity, beta):
        super().__init__(x, y, z, ax_in, fig)
        self.ax_out = ax_out
        self.ax10 = ax10
        self.ax12 = ax12
        self.snr = snr
        self.threshold = threshold
        self.v, self.b = velocity, beta

    def __call__(self, event1, event2):
        super().__call__(event1, event2)
        self.ax10.cla()
        self.ax12.cla()
        self.ax_out.cla()
        self.ax_out.set_title('co_signal vs depo')
        self.ax_out.set_xlabel('co_signal')
        self.ax_out.set_ylabel('depo')
        self.selected_depo = self.area.flatten()
        self.selected_snr = self.snr[self.mask].flatten()
        self.selected_v = self.v[self.mask].flatten()
        self.selected_b = self.b[self.mask].flatten()
        self.ax_out.plot(self.selected_snr, self.selected_depo, '.')
        if self.threshold is not None:
            self.ax_out.axvline(self.threshold)
        self.area_aerosol = span_aerosol(
            self.selected_snr, self.selected_depo, self.ax_out, self.canvas,
            self.ax10, self.ax12, self.selected_v, self.selected_b
        )
        self.canvas.draw()


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
            f'selected area mean is {area_mean:.7f} \n calculated threshold is {self.threshold:.7f}')
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
        self.ax_depo.set_xlabel(f"Depo value colored by SNR at time {time_point:.3f}")
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

        self.ax_depo.plot(self.time, self.depo_max_snr, '-', label='depo at maxSNR')
        self.ax_depo.plot(self.time, self.depo_max_snr1, '--',
                          alpha=0.3, label='depo at maxSNR-1')
        self.ax_depo.set_ylabel('Depo value')
        self.ax_depo.set_title('Depo time series in selected area', weight='bold')
        self.ax_depo.legend()

        self.ax_snr.plot(self.time, self.max_snr, '-', label='maxSNR')
        self.ax_snr.plot(self.time, self.max_snr1, '--', label='maxSNR-1')
        self.ax_snr.legend()
        self.ax_snr.set_xlabel('Time (h)')
        self.ax_snr.set_ylabel('SNR')
        self.ax_snr.set_title('SNR time series in selected area', weight='bold')

        sns.kdeplot(self.max_snr, label='maxSNR', ax=self.ax_hist_snr)
        sns.kdeplot(self.max_snr1, label='maxSNR-1',
                    linestyle="--", ax=self.ax_hist_snr)
        self.ax_hist_snr.set_title('SNR distribution', weight='bold')

        sns.kdeplot(self.depo_max_snr, label='depo at maxSNR', ax=self.ax_hist_depo)
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
    data_indices = [i for i, s in enumerate(data) if date in s.replace(data_folder, '')]
    assert len(data_indices) == 1, 'There are more than 1 data have that date'
    return data_indices[0]


def m_km_ticks():
    '''
    Modify ticks from m to km
    '''
    return FuncFormatter(lambda x, pos: f'{x/1000:.1f}')


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


def aggregate_data(nc_path, noise_path,
                   start_date, end_date, height_lim=100,
                   snr_mul=3,
                   cloud_thres=10**-4.5, cloud_buffer=2,
                   attenuation=True, positive_depo=True,
                   co_cross=False):
    '''
    Aggerate and preprocess data
    date format: '%Y-%m-%d'
    '''
    import glob
    date_range = pd.date_range(start=start_date,
                               end=end_date).strftime('%Y%m%d')

    data_list = glob.glob(nc_path + '/*.nc')
    noise_list = glob.glob(noise_path + '/*_noise.csv')

    depo_raw = []
    v_raw = []
    beta_raw = []
    date_raw = []
    range_raw = {}
    time_raw = {}

    if co_cross:
        co_raw = []
        cross_raw = []

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

        noise_csv = [noise_file for noise_file in noise_list
                     if df.filename in noise_file]

        assert noise_csv, "Missing noise csv for " + df.filename
        noise_csv = noise_csv[0]
        noise = pd.read_csv([noise_file for noise_file in noise_list
                             if df.filename in noise_file][0],
                            usecols=['noise'])
        noise_threshold = 1 + snr_mul * np.std(noise['noise'])
        df.filter(variables=['beta_raw', 'v_raw', 'depo_raw'],
                  ref='co_signal', threshold=noise_threshold)

        if attenuation:
            df.filter_attenuation(variables=['beta_raw', 'v_raw', 'depo_raw'],
                                  ref='beta_raw',
                                  threshold=cloud_thres, buffer=cloud_buffer)

        for depo_value in df.data['depo_raw'][:, :height_lim].ravel():
            depo_raw.append(depo_value)
        for v_value in df.data['v_raw'][:, :height_lim].ravel():
            v_raw.append(v_value)
        b = df.data['beta_raw'][:, :height_lim].ravel()
        for beta_value in b:
            beta_raw.append(beta_value)
        for date_value in np.repeat(date, len(b)):
            date_raw.append(date_value)
        if co_cross:
            for co_value in df.data['co_signal'][:, :height_lim].ravel():
                co_raw.append(co_value)
            for cross_value in df.data['cross_signal'][:, :height_lim].ravel():
                cross_raw.append(cross_value)

        range_raw[date] = df.data['range'][:height_lim]
        time_raw[date] = df.data['time']

    # depo_raw = np.array(depo_raw)
    # v_raw = np.array(v_raw)
    # beta_raw = np.array(beta_raw)
    # date_raw = np.array(date_raw)
    # date_raw = date_raw.astype('int')
    if co_cross:
        # co_raw = np.array(co_raw)
        # cross_raw = np.array(cross_raw)
        result = pd.DataFrame({'depo': depo_raw, 'v_raw': v_raw,
                               'beta_raw': beta_raw, 'date': date_raw,
                               'co_signal': co_raw, 'cross_signal': cross_raw})
    else:
        result = pd.DataFrame({'depo': depo_raw, 'v_raw': v_raw,
                               'beta_raw': beta_raw, 'date': date_raw})
    if positive_depo:
        result.loc[result['depo'] < 0, 'depo'] = np.nan
    return result, time_raw, range_raw
