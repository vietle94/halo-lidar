# %% Load modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import netcdf

# %% get data


def getdata(path, pattern=""):
    ''' Get .nc files from a path,
    it will return a generator.
    I will add pattern later to filter file if needed
    '''
    import glob
    data_list = glob.glob(path + "*.nc")
    return (data for data in data_list)


data = getdata("C:/Users/LV/Downloads/FMI/halo/53/depolarization/")

# %%
df = netcdf.NetCDFFile(next(data), 'r')

# %%
# Data attribute
df_att = list(df.variables.keys())
print(df_att)

# Data attributes with only one value
{name: df.variables[name].getValue() for name in df_att if df.variables[name].shape == ()}

# Data attributes with more than one value
data = {name: df.variables[name][:] for name in df_att if df.variables[name].shape != ()}
data.keys()
# Shape of each data
{key: item.shape for key, item in data.items()}

# Look at each attribute
data.get('time')

data.get('co_signal')
