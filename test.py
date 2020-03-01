# %% Load modules
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import halo_data as hd

# %%
# data = hd.getdata("C:/Users/LV/OneDrive - University of Helsinki/FMI/halo/53/depolarization/")
data = hd.getdata("G:/OneDrive - University of Helsinki/FMI/halo/53/depolarization/")
# data = hd.getdata(r'G:\OneDrive - University of Helsinki\FMI\halo\53\depolarization')

# %% get data
file_name = next(data)
df = hd.halo_data(file_name)
df.info
df.full_data
df.full_data_names
df.data
df.data_names

# %% Histogram of an area in SNR plot
%matplotlib qt
fig, ax = plt.subplots(1, 2)
p = ax[0].pcolormesh(df.data['time'],
                     df.data['range'],
                     df.data['co_signal'].transpose(),
                     cmap='jet', vmin=0.995, vmax=1.005)
area = hd.area_histogram(ax, fig, df.data['time'],
                         df.data['range'],
                         df.data['co_signal'].transpose(),
                         hist=False)
fig.colorbar(p, ax=ax[0])
# %% Calculate threshold
noise = area.area - 1
threshold = 1 + np.nanmean(noise) + np.nanstd(noise) * 2

threshold


# %%
df.filter(variables=['beta_raw', 'v_raw', 'cross_signal', 'depo_raw'],
          ref='co_signal', threshold=threshold)

df.filter(variables=['cross_signal_averaged', 'depo_averaged_raw'],
          ref='co_signal_averaged', threshold=np.percentile(df.data['co_signal_averaged'], 99))

# %%
# Plot data
%matplotlib inline
df.plot(
    variables=['beta_raw', 'v_raw', 'cross_signal', 'depo_raw', 'co_signal',
               'cross_signal_averaged', 'depo_averaged_raw', 'co_signal_averaged'], ncol=2, size=(20, 15))


# %% Summary
df.describe()
