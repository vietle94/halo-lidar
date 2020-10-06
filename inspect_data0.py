import halo_data as hd
import numpy as np
import matplotlib.pyplot as plt
import glob
%matplotlib qt

# %%
data = hd.getdata('F:/halo/33/depolarization')

# %%
date = '20160314'
file = [file for file in data if date in file][0]
df = hd.halo_data(file)

df.filter_height()
df.unmask999()
df.depo_cross_adj()

# %%
fig, ax = plt.subplots()
ax.pcolormesh(df.data['time'], df.data['range'],
              np.log10(df.data['beta_raw']).T, cmap='jet', vmin=-8, vmax=-4)
p = hd.area_select(df.data['time'], df.data['range'],
                   df.data['depo_raw'].T, ax_in=ax, fig=fig)

# %%
df.filter(variables=['depo_adj', 'depo_adj_sd'],
          ref='co_signal',
          threshold=1 + 3*df.snr_sd)

dep = np.nanmean(df.data['depo_adj'].T[p.mask])
dep_sd = np.nanmean(df.data['depo_adj_sd'].T[p.mask])

print(dep, dep_sd)
