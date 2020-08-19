import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

# %%
df = pd.read_excel('F:/elevated_aerosol.xlsx')
hyy = df['hyy'].dropna()
uto = df['uto '].dropna()
kump = df['kump'].dropna()
vehm = df['vehm'].dropna()
sod = df['sod'].dropna()

# %%
# Define Ticks
DAYS = ['Sun', 'Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat']
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']

fig, ax = plt.subplots(3, 1, figsize=(20, 6))
for (i, val), location in zip(enumerate([uto, kump, vehm]),
                              ['Uto', 'Kumpula', 'Vehmesmaki']):
    start = pd.to_datetime('2018-01-01')
    end = pd.to_datetime('2018-12-31')
    start_sun = start - np.timedelta64((start.dayofweek + 1) % 7, 'D')
    end_sun = end + np.timedelta64(7 - end.dayofweek - 1, 'D')

    num_weeks = (end_sun - start_sun).days // 7
    heatmap = np.zeros((7, num_weeks))
    ticks = {}
    y = np.arange(8) - 0.5
    x = np.arange(num_weeks + 1) - 0.5
    for week in range(num_weeks):
        for day in range(7):
            date = start_sun + np.timedelta64(7 * week + day, 'D')
            if date.day == 1:
                ticks[week] = MONTHS[date.month - 1]
            if date.dayofyear == 1:
                ticks[week] += f'\n{date.year}'
            if start <= date < end:
                heatmap[day, week] = 1 if (date == val).any() else None

    cmap = colors.ListedColormap(['gray', 'whitesmoke', 'tab:blue'])
    mesh = ax[i].pcolormesh(x, y, heatmap, cmap=cmap, edgecolors='grey')

    ax[i].invert_yaxis()

    # Set the ticks.
    ax[i].set_xticks(list(ticks.keys()))
    ax[i].set_xticklabels(list(ticks.values()))
    ax[i].set_yticks(np.arange(7))
    ax[i].set_yticklabels(DAYS)
    ax[i].set_ylim(6.5, -0.5)
    ax[i].set_aspect('equal')
    ax[i].set_title(location, fontsize=15)
# Add color bar at the bottom
cbar_ax = fig.add_axes([0.25, -0.10, 0.5, 0.05])
fig.colorbar(mesh, orientation="horizontal", pad=0.2, cax=cbar_ax)
n = 1 + 1
colorbar = ax[2].collections[0].colorbar
r = colorbar.vmax - colorbar.vmin
colorbar.set_ticks([0.16, 0.5, 0.83])
colorbar.set_ticklabels(['Outofbound', 'No idea', 'Elevated aerosol'])
fig.suptitle('Detected elevated aerosol layer', fontweight='bold', fontsize=25)
fig.subplots_adjust(hspace=0.5)
fig.savefig('Hannah.png', bbox_inches='tight')
