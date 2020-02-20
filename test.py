# %% Load modules
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import halo_data as hd


# %%
data = hd.getdata("C:/Users/LV/OneDrive - University of Helsinki/FMI/halo/53/depolarization/")
# data = hd.getdata("G:/OneDrive - University of Helsinki/FMI/halo/53/depolarization/")
# data = hd.getdata(r'G:\OneDrive - University of Helsinki\FMI\halo\53\depolarization')

# %% get data
file_name = next(data)
df = hd.halo_data(file_name)
df.info
df.full_data
df.full_data_names
df.data
df.data_names


df.filter(variables=['beta_raw', 'v_raw', 'cross_signal', 'depo_raw'],
          ref='co_signal', threshold=np.percentile(df.data['co_signal'], 99))

df.filter(variables=['cross_signal_averaged', 'depo_averaged_raw'],
          ref='co_signal_averaged', threshold=np.percentile(df.data['co_signal_averaged'], 99))

# %%
# Plot data
df.plot(
    variables=['beta_raw', 'v_raw', 'cross_signal', 'depo_raw', 'co_signal',
               'cross_signal_averaged', 'depo_averaged_raw', 'co_signal_averaged'],
    nrow=4, ncol=2, size=(20, 15))


# %% Summary
df.describe()
