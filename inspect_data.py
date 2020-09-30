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

# %%
t = (df.data['time'] > 20) & (df.data['time'] < 24)
range_co = (df.data['range'] > 3200) & (df.data['range'] < 8000)
range_cross = (df.data['range'] > 2000) & (df.data['range'] < 8000)

# %%
co = np.nanmean(df.data['co_signal'][np.ix_(t, range_co)], axis=0)
cross = np.nanmean(df.data['cross_signal'][np.ix_(t, range_cross)], axis=0)
x_co = df.data['range'][range_co]
x_cross = df.data['range'][range_cross]

# %%
a, b, c = np.polyfit(x_co, co, deg=2)
y_co = c + b*df.data['range'] + a*(df.data['range']**2)

a, b, c = np.polyfit(x_cross, cross, deg=2)
y_cross = c + b*df.data['range'] + a*(df.data['range']**2)

# %%
fig, ax = plt.subplots(figsize=(6, 6))

ax.plot(np.nanmean(df.data['co_signal'][t, :], axis=0),
        df.data['range'], '.', label='co_signal', c='red')
ax.plot(y_co, df.data['range'], c='red')
ax.plot(np.nanmean(df.data['cross_signal'][t, :], axis=0),
        df.data['range'], '.', label='cross_signal', c='blue')
ax.plot(y_cross, df.data['range'], c='blue')
ax.legend()
ax.set_xlim([0.9995, 1.003])

fig.tight_layout()

# %%
df.data['co_signal'] = df.data['co_signal']/y_co
df.data['cross_signal'] = df.data['cross_signal']/y_cross

# %%
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
