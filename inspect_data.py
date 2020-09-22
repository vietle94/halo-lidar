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

df.filter(variables=['cross_signal', 'beta_raw'],
          ref='co_signal',
          threshold=1 + 3*df.snr_sd)

# %%
fig, ax = plt.subplots()
ax.pcolormesh(df.data['time'], df.data['range'],
              np.log10(df.data['beta_raw']).T, cmap='jet', vmin=-8, vmax=-4)
p = hd.area_select(df.data['time'], df.data['range'],
                   df.data['depo_raw'].T, ax_in=ax, fig=fig)


# %%
# Averaged depo of the whole box
cross = np.nanmean(df.data['cross_signal'].T[p.mask])
co = np.nanmean(df.data['co_signal'].T[p.mask])


print((cross-1)/(co-1))
