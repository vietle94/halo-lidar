import halo_data as hd
import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import pandas as pd
%matplotlib qt

# %%
data = hd.getdata('F:/halo/33/depolarization')
image_folder = 'F:/HYSPLIT/new/'
Path(image_folder).mkdir(parents=True, exist_ok=True)

# %%
date = '20160508'
file = [file for file in data if date in file][0]
df = hd.halo_data(file)

df.filter_height()
df.unmask999()

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
fig.savefig(image_folder + df.filename + '.png',
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
save_df.to_csv(image_folder + df.filename + '_mean_profile.csv',
               index=False)

# %%
# %%


class area_select():

    def __init__(self, x, y, z, ax_in, fig):
        self.x, self.y, self.z = x, y, z
        self.ax_in = ax_in
        self.canvas = fig.canvas
        self.fig = fig
        self.selector = RectangleSelector(
            self.ax_in,
            self,
            useblit=True,  # Process much faster,
            interactive=True  # Keep the drawn box on screen
        )

    def __call__(self, event1, event2):
        self.mask = self.inside(event1, event2)
        self.area = self.z[self.mask]
        self.range = self.y[self.maskrange]
        self.time = self.x[self.masktime]
        print(f'Chosen {len(self.area.flatten())} values')

    def inside(self, event1, event2):
        """
        Returns a boolean mask of the points inside the rectangle defined by
        event1 and event2
        """
        self.xcord = [event1.xdata, event2.xdata]
        self.ycord = [event1.ydata, event2.ydata]
        x0, x1 = sorted(self.xcord)
        y0, y1 = sorted(self.ycord)
        self.masktime = (self.x > x0) & (self.x < x1)  # remove bracket ()
        self.maskrange = (self.y > y0) & (self.y < y1)
        return np.ix_(self.maskrange, self.masktime)

# %%


class span_select():

    def __init__(self, x, y, ax_in, canvas, orient='vertical'):
        self.x, self.y = x, y
        self.ax_in = ax_in
        self.canvas = canvas
        self.selector = SpanSelector(
            self.ax_in, self, orient, span_stays=True, useblit=True
        )

    def __call__(self, min, max):
        self.masky = (self.y > min) & (self.y < max)
        self.selected_x = self.x[self.masky]
        self.selected_y = self.y[self.masky]

# %%


class span_aerosol(span_select):

    def __init__(self, x, y, ax_in, canvas, orient, cross, range_aerosol, df,
                 ax3, ax4):
        super().__init__(x, y, ax_in, canvas, orient)
        self.range_aerosol, self.df = range_aerosol, df
        self.co, self.range, self.cross = x, y, cross
        self.co_all = np.array([])
        self.cross_all = np.array([])
        self.range_all = np.array([])
        self.ax3, self.ax4 = ax3, ax4

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
        # print(f'you choose the background area between: \n' +
        #       f'{self.min/1000:.2f}km and {self.max/1000:.2}km')

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

        co_corrected = self.co/y_co
        self.co_corrected = co_corrected
        cross_corrected = self.cross/y_cross
        cross_sd_background = np.nanstd(cross/y_cross_background)
        co_sd_background = np.nanstd(co/y_co_background)
        self.sigma_co, self.sigma_cross = co_sd_background, cross_sd_background
        bleed = self.df.bleed_through_mean
        sigma_bleed = self.df.bleed_through_sd

        self.cross_corrected = (cross_corrected - 1) - \
            self.df.bleed_through_mean * (co_corrected - 1) + 1

        cross_sd_background_bleed = np.sqrt(
            self.sigma_cross**2 +
            ((bleed * (co_corrected - 1))**2 *
             ((sigma_bleed/bleed)**2 +
              (self.sigma_co/(co_corrected - 1))**2))
        )

        self.depo_corrected = (cross_corrected - 1) / \
            (co_corrected - 1)

        self.depo_corrected_sd = np.sqrt(
            (self.depo_corrected)**2 *
            (
                (cross_sd_background_bleed/(cross_corrected - 1))**2 +
                (self.sigma_co/(co_corrected - 1))**2
            ))

        self.depo_corrected[co_corrected < 1 + 3*co_sd_background] = np.nan
        self.depo_corrected_sd[co_corrected < 1 + 3*co_sd_background] = np.nan
        self.final_depo = np.nanmean(self.depo_corrected[self.range_aerosol])
        self.final_depo_sd = np.nanmean(self.depo_corrected_sd[self.range_aerosol])
        print(f'Depo: {self.final_depo:.3f} with std: {self.final_depo_sd:.3f}')
        self.ax3.plot(self.co_corrected,
                      self.range, '.', label='co_signal', c='red')
        self.ax3.plot(self.cross_corrected,
                      self.range, '.', label='cross_signal', c='blue')
        self.ax3.set_xlim([0.9995, 1.003])
        self.ax3.legend()
        self.ax3.set_title('Corrected co and cross')
        self.ax3.set_xlabel('SNR')
        self.ax4.errorbar(self.depo_corrected,
                          self.range, xerr=self.depo_corrected_sd,
                          errorevery=1, elinewidth=0.5, fmt='.')
        self.ax4.set_xlabel('Depolarization ratio')
        self.ax4.set_title('Averaged depo profile')
        self.ax4.set_xlim([-0.1, 0.5])
        self.canvas.draw()
        self.depo_select = span_depo_select(self.depo_corrected, self.range,
                                            self.ax4, self.canvas, 'vertical',
                                            self.depo_corrected_sd)


# %%
class area_aerosol(area_select):

    def __init__(self, x, y, z, ax_in, fig, ax2, df, ax3, ax4):
        super().__init__(x, y, z, ax_in, fig)
        self.df = df
        self.co = self.df.data['co_signal']
        self.cross = self.df.data['cross_signal']
        self.ax2, self.ax3, self.ax4 = ax2, ax3, ax4

    def __call__(self, event1, event2):
        super().__call__(event1, event2)
        self.ax2.cla()
        self.ax3.cla()
        self.ax4.cla()
        self.t = (self.x >= self.xcord[0]) & (self.x <= self.xcord[1])
        self.range_aerosol = (self.y > self.ycord[0]) & (self.y < self.ycord[1])
        self.co_mean_profile = np.nanmean(self.co[self.t, :], axis=0)
        self.cross_mean_profile = np.nanmean(self.cross[self.t, :], axis=0)
        self.ax2.plot(self.co_mean_profile,
                      self.y, '.', label='co_signal', c='red')
        self.ax2.plot(self.cross_mean_profile,
                      self.y, '.', label='cross_signal', c='blue')
        self.ax2.yaxis.set_major_formatter(m_km_ticks())
        self.ax2.legend()
        self.ax2.set_xlim([0.9995, 1.003])
        self.ax2.set_title('Co and cross profile')
        self.ax2.set_ylabel('Height (km)')
        self.ax2.set_xlabel('SNR')
        self.axapply = self.fig.add_axes([0.7, 0.005, 0.1, 0.025])
        self.axfit = self.fig.add_axes([0.81, 0.005, 0.1, 0.025])
        self.bfit = Button(self.axfit, 'Fit')
        self.bapply = Button(self.axapply, 'Apply')
        self.canvas.draw()
        self.span_aerosol = span_aerosol(self.co_mean_profile, self.y, self.ax2,
                                         self.canvas, 'vertical',
                                         self.cross_mean_profile,
                                         self.range_aerosol,
                                         self.df, self.ax3, self.ax4)
        self.bfit.on_clicked(self.span_aerosol.fit)
        self.bapply.on_clicked(self.span_aerosol.apply)


# %%
data = hd.getdata('F:/halo/46/depolarization')
image_folder = 'F:/HYSPLIT/'
Path(image_folder).mkdir(parents=True, exist_ok=True)

# %%
date = '20180415'
file = [file for file in data if date in file][0]
df = hd.halo_data(file)

df.filter_height()
df.unmask999()

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
ax1.set_xticks(np.arange(24))
ax1.set_ylim([0, None])
ax1.set_ylabel('Height (km)')
ax1.yaxis.set_major_formatter(hd.m_km_ticks())
fig.tight_layout()
fig.subplots_adjust(bottom=0.1, hspace=0.3)
p = hd.area_aerosol(df.data['time'], df.data['range'],
                    df.data['depo_raw'].T, ax_in=ax1,
                    fig=fig, ax2=ax2, df=df, ax3=ax3, ax4=ax4)

# %%
dep0_1 = p.span_aerosol.depo_corrected[df.data['range'] < 1500]

# %%
fig.savefig(image_folder + df.filename + '45.png',
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
save_df.to_csv(image_folder + df.filename + '_mean_profile45.csv',
               index=False)

# %%

list = glob.glob(image_folder + '*.csv')
list
all_profile = pd.concat([pd.read_csv(x) for x in list])
all_profile = all_profile[all_profile['range'] < 2000]
all_profile

# %%
plt.hist(all_profile['depo'])
