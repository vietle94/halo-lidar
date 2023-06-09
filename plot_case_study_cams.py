import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as feature
import xarray as xr
import glob

# %%
file_dir = r'F:\halo\paper\CAMS/'
files = glob.glob(file_dir + 'Finland_2018-04-14_15/*.nc')
df = xr.open_dataset(files[1])


# %%
fig, ax = plt.subplots(figsize=(9, 6), subplot_kw={'projection': ccrs.PlateCarree()},
                       constrained_layout=True)
df['duaod550'][3].plot(transform=ccrs.PlateCarree(), ax=ax,
                       cmap='Oranges')
ax.scatter(24.29, 61.84, c='blue', transform=ccrs.PlateCarree())
ax.annotate('Hyytiälä', (24.39, 61.84), transform=ccrs.PlateCarree())
ax.scatter(21.37, 59.77, c='blue', transform=ccrs.PlateCarree())
ax.annotate('Utö', (21.47, 59.77), transform=ccrs.PlateCarree())
ax.add_feature(feature.BORDERS)
ax.add_feature(feature.COASTLINE)
ax.add_feature(feature.LAND)
ax.add_feature(feature.OCEAN)
ax.set_xlim(right=32)
ax.set_ylim([59, 70.5])
gl = ax.gridlines(draw_labels=True)
gl.right_labels = gl.top_labels = False
fig.savefig(file_dir + 'aod.png', bbox_inches='tight', dpi=600)

# %%
files = glob.glob(file_dir + 'dust_0.9_20/*.nc')
df = xr.open_dataset(files[1])

# %%
fig, ax = plt.subplots(figsize=(9, 6), subplot_kw={'projection': ccrs.PlateCarree()},
                       constrained_layout=True)
df['aermr06'][6].plot(transform=ccrs.PlateCarree(), ax=ax,
                      cmap='Oranges')
ax.scatter(24.29, 61.84, c='blue', transform=ccrs.PlateCarree())
ax.annotate('Hyytiälä', (24.39, 61.84), transform=ccrs.PlateCarree())
ax.scatter(21.37, 59.77, c='blue', transform=ccrs.PlateCarree())
ax.annotate('Utö', (21.47, 59.77), transform=ccrs.PlateCarree())
ax.add_feature(feature.BORDERS)
ax.add_feature(feature.COASTLINE)
ax.add_feature(feature.LAND)
ax.add_feature(feature.OCEAN)
ax.set_xlim(right=32)
ax.set_ylim([59, 70.5])
gl = ax.gridlines(draw_labels=True)
gl.right_labels = gl.top_labels = False

# %%
df['aermr06'].shape
