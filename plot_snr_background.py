import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as dates
import string

# %%
site_names = ['Uto-32', 'Uto-32XR', 'Hyytiala-33',
              'Hyytiala-46', 'Vehmasmaki-53', 'Sodankyla-54']
snr = 'F:/halo/paper/snr_background/'
integration = 'F:/halo/paper/integration_time/'

# %%

snr = 'F:/halo/paper/snr_background/'
integration = 'F:/halo/paper/integration_time/'
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for site in site_names:
    df1 = pd.read_csv(snr + site + '.csv')
    df2 = pd.read_csv(integration + site + '.csv')
    df = df1.merge(df2, how='left')
    df['time'] = pd.to_datetime(df['time'])
    axes[0].plot(df['time'], df['noise'], '.', label=site,
                 markeredgewidth=0.0)
    axes[0].set_ylabel('$\sigma_{SNR}$')
    if site == 'Uto-32XR':
        axes[1].plot(df['time'], df['noise'] * np.sqrt(df['integration_time']*10000),
                     '.', label=site,
                     markeredgewidth=0.0)
    else:
        axes[1].plot(df['time'], df['noise'] * np.sqrt(df['integration_time']*15000),
                     '.', label=site,
                     markeredgewidth=0.0)
    axes[1].set_ylabel('$\sigma_{SNR}$ x $\sqrt{N}$')
    axes[0].set_ylim([0, 0.0025])
for ax in axes.flatten():
    ax.xaxis.set_major_locator(dates.MonthLocator(6))
    ax.set_xlabel('Time')
    ax.grid()
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=6)
for n, ax in enumerate(axes.flatten()):
    ax.text(-0.0, 1.05, '(' + string.ascii_lowercase[n] + ')',
            transform=ax.transAxes, size=12)
fig.subplots_adjust(wspace=0.4, bottom=0.2)
fig.savefig('F:/halo/paper/figures/background_snr/background_snr.png', dpi=300,
            bbox_inches='tight')

# %%
fig, axes = plt.subplots(1, 1, figsize=(6, 4))
for site in site_names:
    df1 = pd.read_csv(snr + site + '.csv')
    df2 = pd.read_csv(integration + site + '.csv')
    df = df1.merge(df2, how='left')
    df['time'] = pd.to_datetime(df['time'])

    if site == 'Uto-32XR':
        axes.plot(df['time'], df['noise'] * np.sqrt(df['integration_time']*10000),
                  '.', label=site,
                  markeredgewidth=0.0)
    else:
        axes.plot(df['time'], df['noise'] * np.sqrt(df['integration_time']*15000),
                  '.', label=site,
                  markeredgewidth=0.0)
    axes.set_ylabel('$\sigma_{SNR}$ x $\sqrt{N}$')
    # axes.set_ylim([0, 0.0025])

axes.xaxis.set_major_locator(dates.MonthLocator(6))
# axes.set_xlabel('Time')
axes.grid()
handles, labels = axes.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3)

fig.tight_layout(rect=[0, 0.1, 1, 0.95])
fig.savefig('F:/halo/paper/figures/background_snr/background_snr_poster.png', dpi=300,
            bbox_inches='tight')
