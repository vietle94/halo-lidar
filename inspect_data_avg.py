import halo_data as hd
import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import pandas as pd
from scipy.ndimage import uniform_filter1d
%matplotlib qt

# %%
data = hd.getdata('F:/halo/33/depolarization')
image_folder = 'F:/HYSPLIT/'
Path(image_folder).mkdir(parents=True, exist_ok=True)

# %%
date = '20160508'
file = [file for file in data if date in file][0]
df = hd.halo_data(file)

df.filter_height()
df.unmask999()
for i in ['beta_raw', 'co_signal', 'cross_signal']:
    df.data[i] = uniform_filter1d(df.data[i], size=10, axis=1)

# %%
fig = plt.figure(figsize=(18, 9))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(234)
ax3 = fig.add_subplot(235, sharey=ax2)
ax4 = fig.add_subplot(236, sharey=ax2)
c = ax1.pcolormesh(df.data['time'], df.data['range'],
                   np.log10(df.data['beta_raw']).T, cmap='jet', vmin=-8, vmax=-4)
cbar = fig.colorbar(c, ax=ax1, fraction=0.01)
cbar.ax.set_ylabel('Beta', rotation=90)
cbar.ax.yaxis.set_label_position('left')
ax1.set_title(df.filename, weight='bold', size=22)
ax1.set_xlabel('Time (h)')
ax1.set_xlim([0, 24])
ax1.set_ylim([0, None])
ax1.set_ylabel('Height (km)')
ax1.yaxis.set_major_formatter(hd.m_km_ticks())
fig.tight_layout()
fig.subplots_adjust(bottom=0.1, hspace=0.3)
p = hd.area_aerosol(df.data['time'], df.data['range'],
                    df.data['depo_raw'].T, ax_in=ax1,
                    fig=fig, ax2=ax2, df=df, ax3=ax3, ax4=ax4)

# %% save result
fig.savefig(image_folder + df.filename + '_h.png',
            bbox_inches='tight')
# save profile as csv
save_df = pd.DataFrame({'range': p.span_aerosol.range,
                        'co_signal': p.span_aerosol.co_corrected,
                        'cross_signal': p.span_aerosol.cross_corrected,
                        'depo': p.span_aerosol.depo_corrected,
                        'depo_sd': p.span_aerosol.depo_corrected_sd,
                        'co_sd': p.span_aerosol.sigma_co,
                        'cross_sd': p.span_aerosol.sigma_cross,
                        'date': df.date,
                        'location': df.location})
save_df.to_csv(image_folder + df.filename + '_mean_profile_h.csv',
               index=False)
