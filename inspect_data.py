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
fig, ax = plt.subplots()
ax.pcolormesh(df.data['time'], df.data['range'],
              np.log10(df.data['beta_raw']).T, cmap='jet', vmin=-8, vmax=-4)
p = hd.area_select(df.data['time'], df.data['range'],
                   df.data['depo_raw'].T, ax_in=ax, fig=fig)
# %%
t = (df.data['time'] >= p.xcord[0]) & (df.data['time'] <= p.xcord[1])
range_aerosol = (df.data['range'] > p.ycord[0]) & (df.data['range'] < p.ycord[1])

# %%
fig, ax = plt.subplots(figsize=(6, 6))

ax.plot(np.nanmean(df.data['co_signal'][t, :], axis=0),
        df.data['range'], '.', label='co_signal', c='red')
# ax.plot(y_co, df.data['range'], c='red')
ax.plot(np.nanmean(df.data['cross_signal'][t, :], axis=0),
        df.data['range'], '.', label='cross_signal', c='blue')
# ax.plot(y_cross, df.data['range'], c='blue')
ax.legend()
ax.set_xlim([0.9995, 1.003])

fig.tight_layout()

# %%
##############################################
# Now choose range based on co_signal for the background
range_co = (df.data['range'] > 3200) & (df.data['range'] < 8000)
##############################################

# %%
n = t.size
co = np.nanmean(df.data['co_signal'][np.ix_(t, range_co)], axis=0)
cross = np.nanmean(df.data['cross_signal'][np.ix_(t, range_co)], axis=0)
x_co = df.data['range'][range_co]
x_cross = df.data['range'][range_co]

a, b, c = np.polyfit(x_co, co, deg=2)
y_co = c + b*df.data['range'] + a*(df.data['range']**2)

a, b, c = np.polyfit(x_cross, cross, deg=2)
y_cross = c + b*df.data['range'] + a*(df.data['range']**2)

# %%
# See the result see if the background range and time fit appropriately
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
co_corrected = np.nanmean(df.data['co_signal'][t, :], axis=0)/y_co
cross_corrected = np.nanmean(df.data['cross_signal'][t, :], axis=0)/y_cross
cross_sd_background = np.nanstd(cross_corrected[range_co])
co_sd_background = np.nanstd(co_corrected[range_co])
sigma_co, sigma_cross = co_sd_background, cross_sd_background
bleed = df.bleed_through_mean
sigma_bleed = df.bleed_through_sd

cross_corrected = (cross_corrected - 1) - \
    df.bleed_through_mean * (co_corrected - 1) + 1

cross_sd_background_bleed = np.sqrt(
    sigma_cross**2 +
    ((bleed * (co_corrected - 1))**2 *
     ((sigma_bleed/bleed)**2 +
      (sigma_co/(co_corrected - 1))**2))
)

depo_corrected = (cross_corrected - 1) / \
    (co_corrected - 1)

depo_corrected_sd = np.sqrt(
    (depo_corrected)**2 *
    (
        (cross_sd_background_bleed/(cross_corrected - 1))**2 +
        (sigma_co/(co_corrected - 1))**2
    ))

depo_corrected[co_corrected < 1 + 3*co_sd_background] = np.nan
depo_corrected_sd[co_corrected < 1 + 3*co_sd_background] = np.nan

final_depo = np.nanmean(depo_corrected[range_aerosol])
final_depo_sd = np.nanmean(depo_corrected_sd[range_aerosol])
print(final_depo, final_depo_sd)
