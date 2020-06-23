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
df.inspect_data()