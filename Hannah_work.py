import numpy as np
import halo_data as hd
import matplotlib.pyplot as plt
import glob
import pandas as pd
%matplotlib qt

# %%
data = hd.getdata('F:/halo/46/depolarization')
snr = glob.glob('F:/halo/46/depolarization/snr/*_noise.csv')

date = '20171212'
file = [file for file in data if date in file][0]
df = hd.halo_data(file)
noise_csv = [noise_file for noise_file in snr
             if df.filename in noise_file][0]
noise = pd.read_csv(noise_csv, usecols=['noise'])
thres = 1 + 3 * np.std(noise['noise'])

# %%
df.filter_height()
df.unmask999()

df.filter(variables=['beta_raw', 'v_raw', 'depo_raw'],
          ref='co_signal', threshold=thres)

# %%
fig = plt.figure(figsize=(18, 9))
ax_in = fig.add_subplot(211)
ax1 = fig.add_subplot(234)
ax2 = fig.add_subplot(235)
ax3 = fig.add_subplot(236)

# %%
b = np.log10(df.data['beta_raw']).T
ax_in.pcolormesh(df.data['time'], df.data['range'],
                 b, cmap='jet',
                 vmin=-8, vmax=-4)
hd.area_classification(df.data['time'], df.data['range'],
                       b, ax_in, fig,
                       df.data['v_raw'].T, df.data['depo_raw'].T,
                       ax1, ax2, ax3)
