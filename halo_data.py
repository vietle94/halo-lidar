from scipy.io import netcdf
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector
import seaborn as sns


def getdata(path, pattern=""):
    ''' Get .nc files from a path,
    it will return a generator.
    I will add pattern later to filter file if needed
    '''
    import glob
    data_list = glob.glob(path + "/*.nc")
    return (data for data in data_list)


class halo_data:
    cbar_lim = {'beta_raw': [-8, -4], 'v_raw': [-1, 1],
                'cross_signal': [0.995, 1.005],
                'co_signal': [0.995, 1.005], 'depo_raw': [0, 0.5],
                'depo_averaged_raw': [0, 0.5], 'co_signal_averaged': [0.995, 1.005],
                'cross_signal_averaged': [0.995, 1.005]}

    def __init__(self, path):
        self.full_data = netcdf.NetCDFFile(path, 'r', mmap=False)
        self.full_data_names = list(self.full_data.variables.keys())
        self.info = {name: self.full_data.variables[name].getValue(
        ) for name in self.full_data_names if self.full_data.variables[name].shape == ()}
        self.data = {name: self.full_data.variables[name].data
                     for name in self.full_data_names if self.full_data.variables[name].shape != ()}
        self.data_names = list(self.data.keys())
        self.more_info = self.full_data._attributes

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
        fig, ax = plt.subplots(nrow, ncol, figsize=size)
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
            axi.set_title(var)
            fig.colorbar(p, ax=axi)
        fig.suptitle(self.full_data.filename.split('\\')[1].split('_')[0] + ' - ' +
                     self.more_info['location'].decode("utf-8") + ' - ' +
                     str(self.more_info['systemID']),
                     size=30,
                     weight='bold')

    def filter(self, variables=None, ref=None, threshold=None):
        for var in variables:
            self.data[var] = np.where(
                self.data[ref] > threshold,
                self.data[var],
                float('nan'))

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


class area_histogram(object):

    def __init__(self, ax, fig, x, y, z, hist=False):
        self.ax = ax
        self.hist = hist
        self.canvas = fig.canvas
        self.x, self.y, self.z = x, y, z
        self.mas = np.zeros((self.x.shape[0], self.y.shape[0]), dtype=bool)
        self.selector = RectangleSelector(
            ax[0],
            self,
            useblit=True,  # process much faster,
            interactive=True)  # Keep the drawn box on screen

    def __call__(self, event1, event2):
        mask = self.inside(event1, event2)
        self.area = self.z[mask]
        print(f'Chosen {len(self.z[mask])} values')
        self.ax[1].cla()
        if self.hist:
            self.ax[1].hist(self.area)
        else:
            sns.kdeplot(self.area, ax=self.ax[1])
        self.canvas.draw()

    def inside(self, event1, event2):
        """Returns a boolean mask of the points inside the rectangle defined by
        event1 and event2."""
        x0, x1 = sorted([event1.xdata, event2.xdata])
        y0, y1 = sorted([event1.ydata, event2.ydata])
        maskx = ((self.x > x0) & (self.x < x1))
        masky = ((self.y > y0) & (self.y < y1))
        maskx = np.repeat(maskx.reshape(1, self.x.shape[0]), self.y.shape[0], axis=0)
        masky = np.repeat(masky.reshape(self.y.shape[0], 1), self.x.shape[0], axis=1)

        return maskx * masky
