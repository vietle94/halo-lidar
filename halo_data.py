from scipy.io import netcdf
import matplotlib.pyplot as plt
import numpy as np


def getdata(path, pattern=""):
    ''' Get .nc files from a path,
    it will return a generator.
    I will add pattern later to filter file if needed
    '''
    import glob
    data_list = glob.glob(path + "*.nc")
    return (data for data in data_list)


class halo_data:

    def __init__(self, path):
        self.full_data = netcdf.NetCDFFile(path, 'r')
        self.names = list(self.full_data.variables.keys())
        self.info = {name: self.full_data.variables[name].getValue(
        ) for name in self.names if self.full_data.variables[name].shape == ()}
        self.data = {name: self.full_data.variables[name][:]
                     for name in self.names if self.full_data.variables[name].shape != ()}
        self.data_names = list(self.data.keys())

    @staticmethod
    def plot(data, variables=None, nrow=None, ncol=None, size=None, vmin_max=None):
        fig, ax = plt.subplots(nrow, ncol, figsize=size)
        for i, var in enumerate(variables):
            if var == 'beta_raw':
                val = np.log10(data.data.get(var)).transpose()
            else:
                val = data.data.get(var).transpose()
            p = ax[i].pcolormesh(data.data.get('time'), data.data.get(
                'range'), val, cmap='jet', vmin=vmin_max.get(var)[0], vmax=vmin_max.get(var)[1])
            ax[i].set_title(var)
            fig.colorbar(p, ax=ax[i])

    def filter(self, variables=None, ref=None, threshold=None):
        for var in variables:
            self.data[var] = np.where(self.data[ref] > threshold, self.data[var], float('nan'))

    def summary(self, variables=None):
        pass
