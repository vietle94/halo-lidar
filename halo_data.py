from scipy.io import netcdf
import matplotlib.pyplot as plt
import numpy as np


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
        self.data = {name: self.full_data.variables[name][:]
                     for name in self.full_data_names if self.full_data.variables[name].shape != ()}
        self.data_names = list(self.data.keys())

    def plot(self, variables=None, nrow=None, ncol=None, size=None):
        fig, ax = plt.subplots(nrow, ncol, figsize=size)
        for i, var in enumerate(variables):
            if var == 'beta_raw':
                val = np.log10(self.data.get(var)).transpose()
            else:
                val = self.data.get(var).transpose()
            p = ax[i].pcolormesh(self.data.get('time'), self.data.get(
                'range'), val, cmap='jet', vmin=self.cbar_lim.get(var)[0], vmax=self.cbar_lim.get(var)[1])
            ax[i].set_title(var)
            fig.colorbar(p, ax=ax[i])

    def filter(self, variables=None, ref=None, threshold=None):
        for var in variables:
            self.data[var] = np.where(self.data[ref] > threshold, self.data[var], float('nan'))

    def describe(self):
        import pandas as pd
        pd.set_option('display.float_format', lambda x: '%.5g' % x)
        var_avg = {var: self.data[var].flatten().astype('f8')
                   for var in self.data if self.data[var].ndim != 1 and 'average' in var}
        varn = {var: self.data[var].flatten()
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
