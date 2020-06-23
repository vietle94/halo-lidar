import numpy as np
import halo_data as hd
import matplotlib.pyplot as plt
import glob
import pandas as pd
from pathlib import Path
%matplotlib qt

# %%
data = hd.getdata('F:/halo/46/depolarization')
cross_signal_folder = 'F:/halo/cross_signal'
Path(cross_signal_folder).mkdir(parents=True, exist_ok=True)

# %%
date = '20171212'
file = [file for file in data if date in file][0]
df = hd.halo_data(file)

# %%
df.cross_signal_sd()

# %%
df.cross_signal_sd_save(cross_signal_folder)
