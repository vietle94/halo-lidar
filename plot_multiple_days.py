# %% Load modules
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import halo_data as hd
from pathlib import Path
import csv
%matplotlib qt


# %%
data = hd.getdata('F:/halo/46/depolarization')
image_folder = 'F:/sorandom/'
Path(image_folder).mkdir(parents=True, exist_ok=True)

# %%
date = '20180514'
file = [file for file in data if date in file][0]
df = hd.halo_data(file)

df.filter_height()
df.unmask999()
df.depo_cross_adj()

df.filter(variables=['beta_raw'],
          ref='co_signal',
          threshold=1 + 3*df.snr_sd)

# %%
date2 = int(date) + 1
date2 = str(date2)
file2 = [file for file in data if date2 in file][0]
df2 = hd.halo_data(file2)

df2.filter_height()
df2.unmask999()
df2.depo_cross_adj()

df2.filter(variables=['beta_raw'],
           ref='co_signal',
           threshold=1 + 3*df2.snr_sd)

# %%
beta = np.hstack((df.data['beta_raw'].T, df2.data['beta_raw'].T))
time = np.concatenate((df.data['time'], df2.data['time'] + 24))

# %%
fig, ax = plt.subplots(figsize=(7, 3))
c = ax.pcolormesh(time, df.data['range'],
                  np.log10(beta), vmin=-8, vmax=-4, cmap='jet')
cbar = fig.colorbar(c, ax=ax)
cbar.ax.set_ylabel('Beta [' + df.units.get('beta_raw', None) + ']', rotation=90)
cbar.ax.yaxis.set_label_position('left')
ax.set_title(df.filename, weight='bold', size=18)
ax.set_xlabel('Time (UTC-h)')
ax.set_xlim([0, 48])
ax.set_ylim([90, None])
ax.set_ylabel('Range [km, a.g.l]')
ax.yaxis.set_major_formatter(hd.m_km_ticks())
fig.tight_layout()
fig.savefig(image_folder + df.filename + '_2days_plot.png',
            dpi=150, bbox_inches='tight')
