from scipy.io import netcdf
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import pandas as pd
from pathlib import Path


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
        self.more_info = self.full_data._attributes

        name = [int(self.more_info.get(key)) if key != 'location' else
                self.more_info.get(key).decode("utf-8") for
                key in ['year', 'month', 'day', 'location', 'systemID']]

        self.filename = '-'.join([str(elem).zfill(2) if name in ['month',
                                                                 'day'] else str(elem).zfill(2) for elem in name])

    def meta_data(self, var=None):
        if var is None or not isinstance(var, str):
            print('Hannah, provide one variable as string')
        else:
            return {key: self.full_data.variables[var].__dict__[key]
                    for key in self.full_data.variables[var].__dict__ if key != 'data'}

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

    def snr_filter(self, multiplier=3):
        fig, ax = plt.subplots(1, 2, figsize=(18, 9))
        p = ax[0].pcolormesh(self.data['time'],
                             self.data['range'],
                             self.data['co_signal'].transpose(),
                             cmap='jet', vmin=0.995, vmax=1.005)
        ax[0].yaxis.set_major_formatter(m_km_ticks())
        ax[0].set_title('Choose background noise for SNR')
        ax[0].set_ylabel('Height (km)')
        ax[0].set_xlabel('Time (h)')
        ax[0].set_ylim(bottom=0)
        self.area_snr = area_snr(self.data['time'],
                                 self.data['range'],
                                 self.data['co_signal'].transpose(),
                                 ax[0],
                                 ax[1],
                                 type='kde',
                                 multiplier=multiplier)
        fig.colorbar(p, ax=ax[0])

    def snr_filter_avg(self, multiplier=3):
        fig, ax = plt.subplots(1, 2, figsize=(18, 9))
        p = ax[0].pcolormesh(self.data['time_averaged'],
                             self.data['range'],
                             self.data['co_signal_averaged'].transpose(),
                             cmap='jet', vmin=0.995, vmax=1.005)
        ax[0].yaxis.set_major_formatter(m_km_ticks())
        ax[0].set_title('Choose background noise for SNR averaged')
        ax[0].set_ylabel('Height (km)')
        ax[0].set_xlabel('Time (h)')
        ax[0].set_ylim(bottom=0)
        self.area_snr_avg = area_snr(self.data['time_averaged'],
                                     self.data['range'],
                                     self.data['co_signal_averaged'].transpose(),
                                     ax[0],
                                     ax[1],
                                     type='kde')
        fig.colorbar(p, ax=ax[0])

    def snr_save(self, snr_folder):
        with open(snr_folder + '/' + self.filename + '_noise.csv', 'w') as f:
            noise_shape = self.area_snr.area.flatten().shape
            noise_csv = pd.DataFrame.from_dict({'year': np.repeat(self.more_info['year'], noise_shape),
                                                'month': np.repeat(self.more_info['month'], noise_shape),
                                                'day': np.repeat(self.more_info['day'], noise_shape),
                                                'location': np.repeat(self.more_info['location'].decode('utf-8'), noise_shape),
                                                'systemID': np.repeat(self.more_info['systemID'], noise_shape),
                                                'noise': self.area_snr.area.flatten()})
            noise_csv.to_csv(f, header=f.tell() == 0, index=False)

        with open(snr_folder + '/' + self.filename + '_noise_avg' + '.csv', 'w') as ff:
            noise_avg_shape = self.area_snr_avg.area.flatten().shape
            noise_avg_csv = pd.DataFrame.from_dict({'year': np.repeat(self.more_info['year'], noise_avg_shape),
                                                    'month': np.repeat(self.more_info['month'], noise_avg_shape),
                                                    'day': np.repeat(self.more_info['day'], noise_avg_shape),
                                                    'location': np.repeat(self.more_info['location'].decode('utf-8'), noise_avg_shape),
                                                    'systemID': np.repeat(self.more_info['systemID'], noise_avg_shape),
                                                    'noise': self.area_snr_avg.area.flatten()})
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
                                        snr=self.data['co_signal'].transpose())
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
            'location': self.more_info['location'].decode('utf-8'),
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
                                       snr=self.data['co_signal'].transpose())
        return fig

    def depo_wholeprofile_save(self, fig, depo_folder):
        n_values = self.depo_wp.time.shape[0]
        result = pd.DataFrame.from_dict({
            'year': np.repeat(self.more_info['year'], n_values),
            'month': np.repeat(self.more_info['month'], n_values),
            'day': np.repeat(self.more_info['day'], n_values),
            'location': np.repeat(self.more_info['location'].decode('utf-8'), n_values),
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


class area_select():

    def __init__(self, x, y, z, ax_in):
        self.x, self.y, self.z = x, y, z
        self.ax_in = ax_in
        self.canvas = plt.gcf().canvas
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


class area_snr(area_select):

    def __init__(self, x, y, z, ax_in, ax_snr, type='hist', multiplier=3):
        super().__init__(x, y, z, ax_in)
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
        self.ax_snr.set_title(
            f'selected area mean is {area_mean:.7f} \n calculated threshold is {self.threshold:.7f}')
        print('Calculated threshold is: ', self.threshold)
        self.ax_snr.axvline(area_mean, c='red')
        self.canvas.draw()


class area_timeprofile(area_select):

    def __init__(self, x, y, z, ax_in, ax_snr, ax_depo, snr,
                 ax_snr_label='SNR', ax_depo_label='Depo'):
        super().__init__(x, y, z, ax_in)
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
                 ax_depo, ax_snr,
                 ax_hist_depo, ax_hist_snr,
                 snr):
        super().__init__(x, y, z, ax_in)
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
        self.ax_depo.plot(self.time, self.depo_max_snr1, '--', label='depo at maxSNR-1')
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


def m_km_ticks():
    '''
    Modify ticks from m to km
    '''
    return FuncFormatter(lambda x, pos: f'{x/1000:.1f}')
