import halo_data as hd
import numpy as np
import matplotlib.pyplot as plt
import glob
%matplotlib qt

# %%
data = hd.getdata('F:/halo/46/depolarization')

# %%
date = '20180611'
file = [file for file in data if date in file][0]
df = hd.halo_data(file)

df.filter_height()
df.unmask999()
df.depo_cross_adj()

# %%
# Modify t: time interval
t = 1
n = 1
while True:
    dif = np.median(df.data['time'][n::n] - df.data['time'][:-n:n])
    if t - dif < 0:
        break
    n += 1

# %%
df.average(n)
df.data['depo_averaged'] = hd.ma(df.data['depo_averaged'].T, n=5).T
df.filter(variables=['depo_averaged'],
          ref='co_signal_averaged',
          threshold=1 + 3*df.snr_sd/np.sqrt(n))


# %%
fig, ax = plt.subplots()
ax.pcolormesh(df.data['time_averaged'], df.data['range'],
              df.data['depo_averaged'].T, cmap='jet', vmin=0, vmax=0.5)
p = hd.area_select(df.data['time_averaged'], df.data['range'],
                   df.data['depo_averaged'].T, ax_in=ax, fig=fig)

# %%
p.area
np.nanmean(p.area)
