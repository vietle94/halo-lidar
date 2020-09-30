import halo_data as hd
import numpy as np
import matplotlib.pyplot as plt
import glob
%matplotlib qt

# %%
data = hd.getdata('F:/halo/33/depolarization')

# %%
date = '20160508'
file = [file for file in data if date in file][0]
df = hd.halo_data(file)

df.filter_height()
df.unmask999()
# df.depo_cross_adj()

# # %%
# # Modify t: time interval
# t = 1
# n = 1
# while True:
#     dif = np.median(df.data['time'][n::n] - df.data['time'][:-n:n])
#     if t - dif < 0:
#         break
#     n += 1

# %%
##############################
# Choose time interval unit is hour
##############################
t = (df.data['time'] > 20) & (df.data['time'] < 24)

co = df.data['co_signal'][t, :] - 1
cross = df.data['cross_signal'][t, :] - 1
cro_sd = np.nanstd(cross, axis=0)

# %%
fig, ax = plt.subplots(1, 2, sharey=True, figsize=(12, 6))
ax[0].plot(np.nanmean(co, axis=0), df.data['range'], label='co_SNR')
ax[0].plot(np.nanmean(cross, axis=0), df.data['range'], label='cross_SNR')
ax[0].errorbar(np.nanmean(cross, axis=0)[::10], df.data['range'][::10],
               xerr=cro_sd[::10])
ax[0].legend()
ax[0].axvline(x=0)
ax[0].set_title('Mean cross_signal profile with green errorbars')

ax[1].plot(np.nanmean(co, axis=0), df.data['range'], label='co_SNR')
ax[1].plot(np.nanmean(cross, axis=0), df.data['range'], label='cross_SNR')
ax[1].legend()
ax[1].axvline(x=0)
ax[1].set_xlim([- 0.0005, 0.002])
ax[1].set_title('Zoomed image')
fig.tight_layout()
# fig.savefig('no_bleedthrough.png', bbox_inches='tight')

# %%
fig, ax = plt.subplots(figsize=(12, 6))
p = ax.pcolormesh(df.data['time'], df.data['range'],
                  np.log10(df.data['beta_raw']).T, vmin=-8, vmax=-4, cmap='jet')
fig.colorbar(p, ax=ax)
ax.set_title('beta_raw')
# fig.savefig('beta.png', bbox_inches='tight')
