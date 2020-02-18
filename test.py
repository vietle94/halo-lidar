# %% Load modules
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import halo_data


# %%
# data = halo_data.getdata(
#     "C:/Users/VIETLE/OneDrive - University of Helsinki/FMI/halo/53/depolarization/")
data = halo_data.getdata("G:/OneDrive - University of Helsinki/FMI/halo/53/depolarization/")

# %% get data
df = halo_data.halo_data(next(data))
df.info
df.names
df.data_names

df.filter(variables=['beta_raw', 'v_raw', 'cross_signal',
                     'depo_raw'], ref='co_signal', threshold=1.0035)
# %%
# Plot data
halo_data.halo_data.plot(df, variables=['beta_raw', 'v_raw',
                                        'cross_signal', 'depo_raw'], nrow=4, ncol=1, size=(12, 12),
                         vmin_max={'beta_raw': [-8, -4], 'v_raw': [-1, 1], 'cross_signal': [1, 1.05], 'depo_raw': [0, 1], })
