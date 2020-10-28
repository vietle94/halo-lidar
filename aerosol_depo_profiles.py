import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

# %%
list_files = glob.glob('F:/HYSPLIT/*_mean_profile.csv')

# %%
date = '2018-04-15'
list_df = [file for file in list_files if date in file]
print('\n'.join(list_df))
# %%
row = 1
column = 3
size = (16, 9)
fig, axes = plt.subplots(row, column, figsize=size, sharey=True, sharex=True)
for ax, f in zip(axes.flatten(), list_df):
    df = pd.read_csv(f)
    range = np.arange(105, df.shape[0]*30 + 105, 30)
    ax.errorbar(df['depo'], range, xerr=df['depo_sd'],
                errorevery=1, elinewidth=0.5, fmt='.')
    ax.set_title(f.split('\\')[-1].split('_')[0], weight='bold')
    ax.set_xlabel('Depolarization ratio')
    ax.set_ylabel('Height')
    ax.set_xlim([-0.2, 0.8])
fig.tight_layout()
fig.savefig('F:/HYSPLIT/' + f.split('\\')[-1].split('_')[0][:10] + '.png',
            bbox_inches='tight')
