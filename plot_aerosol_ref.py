import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib qt

# %%
df = pd.read_csv(r'G:\OneDrive - University of Helsinki/depo_wave.csv')

# %%
my_cmap = plt.cm.get_cmap('Blues')
wave_color = {'355nm': my_cmap(0.99), '532nm': my_cmap(0.75), '710nm': my_cmap(0.5),
              '1064nm': my_cmap(0.25), '1565nm': my_cmap(0.1)}

# %%
df['Type'] = ['dust' if 'dust' in x else x for x in df['Type']]
df['Wavelength'] = [int(x[:-2]) if 'nm' in x else x for x in df['Wavelength']]
df['color'] = [wave_color[str(i) + 'nm'] for i in x]

# %%
df = df.sort_values(['Type', 'Wavelength']).reset_index(drop='True')

# %%
fig, ax = plt.subplots()
for i, val in df.iterrows():
    ax.errorbar(i, val['Depol0'], val['Depol1'], fmt='.',
                elinewidth=1, c=val['color'])

# %%
fig, ax = plt.subplots()
ax.errorbar(df[])
val['Depol']
tuple(val['Depol'])

# %%
x = []
y = []
yerr = []
x_type = []
for type, type_lab in zip([dust, pollen, marine, ash, smoke],
                          ['dust', 'pollen', 'marine',
                           'ash',
                           'smoke']):
    for i in type:
        for author, vi in i.items():
            for wa, vii in vi.items():
                x.append(int(wa[:-2]))
                y.append(vii[0])
                yerr.append(vii[1])
                x_type.append(type_lab)

# %%
wave_color_x = [wave_color[str(i) + 'nm'] for i in x]
fig, ax = plt.subplots()
for i_, x_, y_, yerr_, color_ in zip(np.arange(len(x)),
                                     x, y, yerr, wave_color_x):
    ax.errorbar(i_, y_, yerr=yerr_, fmt='.',
                elinewidth=1, c=color_)

# %%
fig, axes = plt.subplots(6, 1, figsize=(5, 15))
axes = axes.flatten()
for type, ax in zip([dust, pollen, marine, ash, smoke], axes):
    for i in type:
        x = []
        y = []
        yerr = []
        for author, vi in i.items():
            for wa, vii in vi.items():
                x.append(int(wa[:-2]))
                y.append(vii[0])
                yerr.append(vii[1])
            author = author.replace(' \n', '')
            ax.errorbar(np.array(x), y, yerr=yerr, fmt='.-',
                        elinewidth=1, label=author)
    ax.legend()
    ax.tick_params(axis='x', bottom=False)
    ax.set_ylabel('Depolarization ratio')
    ax.set_ylim([-0.07, 0.4])
    ax.set_xlim([300, 1600])
    ax.set_xticks([355, 532, 710, 1064, 1565])
    ax.legend(bbox_to_anchor=(1.01, 1.05), loc="upper left")

grp = df.groupby('location2')['depo']
x = 1565 + np.array([-20, -7, 7, 20])
for xx, mean, std, (lab, v) in zip(x, grp.mean(), grp.std(), grp):
    axes[-1].errorbar(xx, mean, std, elinewidth=1, fmt='.-', label=lab)
axes[-1].set_xticks([355, 532, 710, 1064, 1565])
axes[-1].set_xlabel('Wavelength (nm)')
axes[-1].legend(bbox_to_anchor=(1.01, 1.05), loc="upper left")
axes[-1].set_ylabel('Depolarization ratio')
axes[-2].set_ylabel('Depolarization ratio')
axes[-1].set_ylim([-0.07, 0.4])
axes[-1].set_xlim([300, 1600])
axes[-1].tick_params(axis='x', bottom=False)
fig.subplots_adjust(hspace=0.3)
