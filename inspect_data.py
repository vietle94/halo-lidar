import halo_data as hd
import numpy as np
import matplotlib.pyplot as plt
import glob
from matplotlib.widgets import RectangleSelector, SpanSelector
from matplotlib.widgets import Button
%matplotlib qt

# %%
data = hd.getdata('F:/halo/33/depolarization')

# %%
date = '20160508'
file = [file for file in data if date in file][0]
df = hd.halo_data(file)

df.filter_height()
df.unmask999()

# %%
fig, ax = plt.subplots(1, 2)
ax[0].pcolormesh(df.data['time'], df.data['range'],
                 np.log10(df.data['beta_raw']).T, cmap='jet', vmin=-8, vmax=-4)
ax[0].set_title(df.filename)
fig.subplots_adjust(bottom=0.2)
# axapply = fig.add_axes([0.7, 0.05, 0.1, 0.075])
# axfit = fig.add_axes([0.81, 0.05, 0.1, 0.075])
# bfit = Button(axfit, 'Fit')
# bapply = Button(axapply, 'Apply')
p = hd.area_aerosol(df.data['time'], df.data['range'],
                    df.data['depo_raw'].T, ax_in=ax[0], fig=fig, ax2=ax[1], df=df)
# bfit.on_clicked(p.span_aerosol.select.fit)
# bapply.on_clicked(p.span_aerosol.apply)

# %%
p.maskrange
p.cross

# %%
t = (df.data['time'] >= p.xcord[0]) & (df.data['time'] <= p.xcord[1])
range_aerosol = (df.data['range'] > p.ycord[0]) & (df.data['range'] < p.ycord[1])

co_mean_profile = np.nanmean(df.data['co_signal'][t, :], axis=0)
cross_mean_profile = np.nanmean(df.data['cross_signal'][t, :], axis=0)

# %%


class span_select():

    def __init__(self, co, range, ax_in, canvas, cross):
        self.co, self.range, self.cross = co, range, cross
        self.ax_in = ax_in
        self.canvas = canvas
        self.co_all = np.array([])
        self.cross_all = np.array([])
        self.range_all = np.array([])
        self.selector = SpanSelector(
            self.ax_in, self, 'vertical',
            span_stays=True, useblit=True
        )

    def __call__(self, min, max):
        self.min = min
        self.max = max
        self.maskrange = (self.range > self.min) & (self.range < self.max)
        self.selected_co = self.co[self.maskrange]
        self.selected_range = self.range[self.maskrange]
        self.selected_cross = self.cross[self.maskrange]

    def apply(self, event):
        self.co_all = np.append(self.co_all, self.selected_co)
        self.cross_all = np.append(self.cross_all, self.selected_cross)
        self.range_all = np.append(self.range_all, self.selected_range)
        self.ax_in.axhspan(self.min, self.max, alpha=0.5, color='yellow')
        self.canvas.draw()
        print(f'you chose the area between {self.min/1000:.2f}km and {self.max/1000:.2}km')

    def fit(self, event):
        co = self.co_all
        cross = self.cross_all
        x_co = self.range_all

        a, b, c = np.polyfit(x_co, co, deg=2)
        y_co = c + b*self.range + a*(self.range**2)
        y_co_background = c + b*x_co + a*(x_co**2)

        a, b, c = np.polyfit(x_co, cross, deg=2)
        y_cross = c + b*self.range + a*(self.range**2)
        y_cross_background = c + b*x_co + a*(x_co**2)

        self.ax_in.plot(y_co, self.range, c='red')
        self.ax_in.plot(y_cross, self.range, c='blue')

        co_corrected = co_mean_profile/y_co
        self.co_corrected = co_corrected
        cross_corrected = cross_mean_profile/y_cross
        cross_sd_background = np.nanstd(cross/y_cross_background)
        co_sd_background = np.nanstd(co/y_co_background)
        sigma_co, sigma_cross = co_sd_background, cross_sd_background
        bleed = df.bleed_through_mean
        sigma_bleed = df.bleed_through_sd

        self.cross_corrected = (cross_corrected - 1) - \
            df.bleed_through_mean * (co_corrected - 1) + 1

        cross_sd_background_bleed = np.sqrt(
            sigma_cross**2 +
            ((bleed * (co_corrected - 1))**2 *
             ((sigma_bleed/bleed)**2 +
              (sigma_co/(co_corrected - 1))**2))
        )

        depo_corrected = (cross_corrected - 1) / \
            (co_corrected - 1)

        depo_corrected_sd = np.sqrt(
            (depo_corrected)**2 *
            (
                (cross_sd_background_bleed/(cross_corrected - 1))**2 +
                (sigma_co/(co_corrected - 1))**2
            ))

        depo_corrected[co_corrected < 1 + 3*co_sd_background] = np.nan
        depo_corrected_sd[co_corrected < 1 + 3*co_sd_background] = np.nan

        self.final_depo = np.nanmean(depo_corrected[range_aerosol])
        self.final_depo_sd = np.nanmean(depo_corrected_sd[range_aerosol])
        print(self.final_depo, self.final_depo_sd)


# %%
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(co_mean_profile,
        df.data['range'], '.', label='co_signal', c='red')
ax.plot(cross_mean_profile,
        df.data['range'], '.', label='cross_signal', c='blue')
ax.legend()
ax.set_xlim([0.9995, 1.003])
fig.subplots_adjust(bottom=0.2)
axapply = fig.add_axes([0.7, 0.05, 0.1, 0.075])
axfit = fig.add_axes([0.81, 0.05, 0.1, 0.075])
bfit = Button(axfit, 'Fit')
bapply = Button(axapply, 'Apply')
ax.set_title(df.filename + f' from {p.xcord[0]:.2f} to {p.xcord[1]:.2f} UTC')
select = hd.span_select(co_mean_profile, df.data['range'], ax, fig.canvas,
                        cross_mean_profile)
# bfit.on_clicked(select.fit)
# bapply.on_clicked(select.apply)
# dir(select)
