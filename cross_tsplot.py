import matplotlib.pyplot as plt
import glob
import pandas as pd

# %%
cross_signal_folder = 'F:/halo/cross_signal'
cross = pd.concat([pd.read_csv(f) for f in
                   glob.glob(cross_signal_folder + '/*.csv')])
cross['date'] = pd.to_datetime(cross[['year', 'month', 'day']])

# %%
fig, axes = plt.subplots(4, 2, figsize=(16, 9))
for (key, grp), ax in zip(cross.groupby(['systemID']),
                          axes.flatten()):
    temp = grp.groupby('date').std()
    ax.plot(temp['noise'], '.')
    ax.set_title(key)
    ax.set_ylabel('Std of cross_signal')

fig.subplots_adjust(hspace=0.5)
fig.savefig('cross_ts.png', dpi=150, bbox_inches='tight')
