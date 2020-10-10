import halo_data as hd
import numpy as np
import matplotlib.pyplot as plt
import glob
from matplotlib.widgets import RectangleSelector, SpanSelector
from matplotlib.widgets import Button
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
fig, ax = plt.subplots(1, 2, sharey=True)
ax[0].pcolormesh(df.data['time'], df.data['range'],
                 np.log10(df.data['beta_raw']).T, cmap='jet', vmin=-8, vmax=-4)
ax[0].set_title(df.filename)
ax[0].yaxis.set_major_formatter(hd.m_km_ticks())
fig.subplots_adjust(bottom=0.2)
p = hd.area_aerosol(df.data['time'], df.data['range'],
                    df.data['depo_raw'].T, ax_in=ax[0],
                    fig=fig, ax2=ax[1], df=df)
