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

# %%
# Choose careful the area of interest
fig, ax = plt.subplots()
ax.pcolormesh(df.data['time'], df.data['range'],
              np.log10(df.data['beta_raw']).T, cmap='jet', vmin=-8, vmax=-4)
p = hd.area_select(df.data['time'], df.data['range'],
                   df.data['depo_raw'].T, ax_in=ax, fig=fig)

# %%
cross = np.nanmean(df.data['cross_signal'].T[p.mask])
co = np.nanmean(df.data['co_signal'].T[p.mask])
sigma_cross = sigma_co = df.snr_sd/np.sqrt(p.mask[0].size * p.mask[1].size)
bleed = df.bleed_through_mean
sigma_bleed = df.bleed_through_sd
cross_corrected = (cross-1) - \
    bleed * (co-1) + 1
cross_sd = np.sqrt(
    sigma_cross**2 +
    ((bleed * (co-1))**2 *
     ((sigma_bleed/bleed)**2 +
      (sigma_co/(co-1))**2))
)

depo_corrected = (cross_corrected - 1) / \
    (co - 1)

depo_corrected_sd = np.sqrt(
    (depo_corrected)**2 *
    (
        (cross_sd/(cross_corrected - 1))**2 +
        (sigma_co/(co - 1))**2
    ))

# %%
dep = np.nanmean(depo_corrected[co > 1+3*sigma_co])
dep_sd = np.nanmean(depo_corrected_sd[co > 1+3*sigma_co])
print(f'depo: {dep:.5f}, depo sd: {dep_sd:.5f}')
