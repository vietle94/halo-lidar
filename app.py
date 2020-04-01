
from PyQt5.QtWidgets import (QApplication, QMessageBox, QMainWindow,
                             QSizePolicy, QGridLayout,
                             QAction, QFileDialog, QLabel, QWidget,
                             QTableView, QAbstractScrollArea)
from PyQt5.QtCore import *
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.uic import loadUiType
from os.path import dirname, realpath, join
from sys import argv
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
# import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import halo_data as hd
import pandas as pd

from PyQt5 import QtCore

scriptDir = dirname(realpath(__file__))
FROM_MAIN, _ = loadUiType(join(dirname(__file__), "mainwindow.ui"))


class Main(QMainWindow, FROM_MAIN):
    def __init__(self, parent=FROM_MAIN):

        super(Main, self).__init__()

        QMainWindow.__init__(self)
        self.setupUi(self)
        self.toolbar()
        self.menubar()
        self.listWidget.addItem("Add folder")

        self.mycanvas = myCanvas()
        self.mycanvas.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.mycanvas.setFocus()
        # self.mycanvas.draw()
        self.frame = QGridLayout(self.Plotframe)
        self.frame.addWidget(self.mycanvas, 0, 0, 1, -6)
        self.frame.setContentsMargins(10, 0, 0, 0)
        self.frame.setSpacing(5)

        self.plt_toolbar = NavigationToolbar(self.mycanvas, self)
        self.frame.addWidget(self.plt_toolbar, 1, 0)
        # self.plt_toolbar.setStyleSheet("QToolBar { border: 0px }")

        self.savepath_lab = QLabel('Please choose saving folder ' + " "*70, self)
        self.menuBar.setCornerWidget(self.savepath_lab)

        self.load_status = QLabel("", self)
        self.frame.addWidget(self.load_status, 1, 1)
        self.pre_status = QLabel("", self)
        self.frame.addWidget(self.pre_status, 1, 2)
        self.snr_status = QLabel("", self)
        self.frame.addWidget(self.snr_status, 1, 3)
        self.depo_save_status = QLabel("", self)
        self.frame.addWidget(self.depo_save_status, 1, 4)
        self.fmi_label = QLabel(self)
        self.fmi_pix = QPixmap('icons/fmi.png')
        self.fmi_label.setPixmap(self.fmi_pix.scaledToHeight(40))
        self.frame.addWidget(self.fmi_label, 1, 5)

        self.nextbutton.clicked.connect(self.next_list)

        # self.fmi_label.resize(50, 50)
        # self.depo_status = QLabel("", self)
        # self.frame.addWidget(self.depo_status, 1, 4)

        # self.Qe = 0
        # self.quit = 0
        # self.p = 0

    def menubar(self):

        self.actionSetdataFolder.setShortcut('Ctrl+O')
        self.actionSetdataFolder.setStatusTip('Open folder')
        self.actionSetdataFolder.triggered.connect(self.Setdata)

        self.actionSetsaveFolder.setShortcut('Ctrl+S')
        self.actionSetsaveFolder.setStatusTip('Set save folder')
        self.actionSetsaveFolder.triggered.connect(self.Setsave)

        self.actionSettings.setShortcut('Ctrl+T')
        self.actionSettings.setStatusTip('Settings')
        self.actionSettings.triggered.connect(self.Settings)

        self.actionAbout_as.setShortcut('Ctrl+T')
        self.actionAbout_as.setStatusTip('Settings')
        self.actionAbout_as.triggered.connect(self.About)

        self.actionExit.setShortcut('Ctrl+Q')
        self.actionExit.setStatusTip('Quit')
        self.actionExit.triggered.connect(self.Quit)

    def Setdata(self):
        self.dataPath = QFileDialog.getExistingDirectory(
            self, "Select directory with .nc files", 'F:/')
        if self.dataPath:
            self.listWidget.clear()
            self.listWidget.addItems(hd.getdata(self.dataPath))

    def Setsave(self):
        self.savePath = QFileDialog.getExistingDirectory(
            self, "Select save directory", 'F:/')
        self.savepath_lab.setText('Results will be saved at ' + self.savePath)
        self.savePath_img = self.savePath + '/img'
        self.savePath_depo = self.savePath + '/depo'
        self.savePath_snr = self.savePath + '/snr'
        for path in [self.savePath_img, self.savePath_depo, self.savePath_snr]:
            Path(path).mkdir(parents=True, exist_ok=True)

    def Settings(self):
        pass

    def About(self):
        pass

    def Quit(self):
        self.close()

    def help(self):
        QMessageBox.critical(self, 'Aide', "Hello This is_ PyQt5 Gui and Matplotlib ")

    def toolbar(self):
        self.loaddata_btn = QAction(QIcon('icons/download.PNG'), 'Load data', self)
        self.loaddata_btn.setEnabled(1)
        self.loaddata_btn.triggered.connect(self.loaddata)

        self.preprocess_btn = QAction(QIcon('icons/preprocess.PNG'), 'Preliminary process', self)
        self.preprocess_btn.setEnabled(1)
        self.preprocess_btn.triggered.connect(self.preprocess)

        self.plotdata_btn = QAction(QIcon('icons/plot.PNG'), 'Plot data', self)
        self.plotdata_btn.setEnabled(1)
        self.plotdata_btn.triggered.connect(self.plotdata)

        self.noiseselect_btn = QAction(QIcon('icons/snr.PNG'), 'Select background noise', self)
        self.noiseselect_btn.setEnabled(1)
        self.noiseselect_btn.triggered.connect(self.noiseselect)

        self.noisefilter_btn = QAction(QIcon('icons/filter.PNG'), 'Filter noise', self)
        self.noisefilter_btn.setEnabled(1)
        self.noisefilter_btn.triggered.connect(self.noisefilter)

        self.plotdatafiltered_btn = QAction(QIcon('icons/plot_filtered.PNG'), 'Plot data', self)
        self.plotdatafiltered_btn.setEnabled(1)
        self.plotdatafiltered_btn.triggered.connect(self.plotdatafiltered)

        self.info_btn = QAction(QIcon('icons/info.PNG'), 'Data info', self)
        self.info_btn.setEnabled(1)
        self.info_btn.triggered.connect(self.info)

        self.depo_timeprofile_btn = QAction(QIcon('icons/depo_time.PNG'), 'Depo time profile', self)
        self.depo_timeprofile_btn.setEnabled(1)
        self.depo_timeprofile_btn.triggered.connect(self.depo_timeprofile)

        self.depo_timesave_btn = QAction(
            QIcon('icons/depo_time_save.PNG'), 'Save depo time profile', self)
        self.depo_timesave_btn.setEnabled(1)
        self.depo_timesave_btn.triggered.connect(self.depo_timesave)

        self.depo_wholeprofile_btn = QAction(
            QIcon('icons/depo_whole.PNG'), 'Depo whole profile', self)
        self.depo_wholeprofile_btn.setEnabled(1)
        self.depo_wholeprofile_btn.triggered.connect(self.depo_wholeprofile)

        self.depo_wholesave_btn = QAction(
            QIcon('icons/depo_whole_save.PNG'), 'Save depo whole profile', self)
        self.depo_wholesave_btn.setEnabled(1)
        self.depo_wholesave_btn.triggered.connect(self.depo_wholesave)

        spacer = QWidget(self)
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        spacer2 = QWidget(self)
        spacer2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.toolbar = self.addToolBar('Actions')
        self.toolbar.setMovable(False)
        self.toolbar.addAction(self.loaddata_btn)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.preprocess_btn)
        self.toolbar.addAction(self.plotdata_btn)
        self.toolbar.addAction(self.noiseselect_btn)
        self.toolbar.addAction(self.noisefilter_btn)
        self.toolbar.addAction(self.plotdatafiltered_btn)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.info_btn)
        self.toolbar.addWidget(spacer2)
        self.toolbar.addAction(self.depo_timeprofile_btn)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.depo_timesave_btn)
        self.toolbar.addWidget(spacer)
        self.toolbar.addAction(self.depo_wholeprofile_btn)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.depo_wholesave_btn)
        # self.toolbar.setFixedSize(500, 50)
        # self.toolbar.setIconSize(QSize(50, 50))

    def next_list(self):
        self.listWidget.setCurrentRow(self.listWidget.currentRow() + 1)
        self.load_status.setText(f'')
        self.pre_status.setText(f'')
        self.snr_status.setText(f'')
        self.depo_save_status.setText(f'')

    def loaddata(self):
        self.halodata = hd.halo_data(self.listWidget.currentItem().text())
        self.load_status.setText(f'Loaded {self.halodata.filename}')

    def preprocess(self):
        self.halodata.unmask999()
        self.halodata.filter_height()
        self.pre_status.setText('Preprocess completed')

    def plotdata(self):
        self.mycanvas.plotdata(variables=['beta_raw', 'v_raw', 'cross_signal',
                                          'depo_raw', 'co_signal',
                                          'cross_signal_averaged', 'depo_averaged_raw',
                                          'co_signal_averaged'],
                               halo=self.halodata)
        self.mycanvas.print_figure(
            self.savePath_img + '/' + self.halodata.filename + '_raw.png',
            dpi=200)

    def plotdatafiltered(self):
        self.mycanvas.plotdata(variables=['beta_raw', 'v_raw', 'cross_signal',
                                          'depo_raw', 'co_signal',
                                          'cross_signal_averaged', 'depo_averaged_raw',
                                          'co_signal_averaged'],
                               halo=self.halodata)
        self.mycanvas.print_figure(
            self.savePath_img + '/' + self.halodata.filename + '_filtered.png',
            dpi=200)

    def noiseselect(self):
        self.mycanvas.snr_filter(
            halo=self.halodata, multiplier=3, multiplier_avg=3)

    def noisefilter(self):
        with open(self.savePath_snr + '/' + self.halodata.filename + '_noise.csv', 'w') as f:
            noise_area = self.mycanvas.area_snr.area.flatten()
            noise_shape = noise_area.shape
            noise_csv = pd.DataFrame.from_dict({'year': np.repeat(self.halodata.more_info['year'], noise_shape),
                                                'month': np.repeat(self.halodata.more_info['month'], noise_shape),
                                                'day': np.repeat(self.halodata.more_info['day'], noise_shape),
                                                'location': np.repeat(self.halodata.more_info['location'].decode('utf-8'), noise_shape),
                                                'systemID': np.repeat(self.halodata.more_info['systemID'], noise_shape),
                                                'noise': noise_area - 1})
            noise_csv.to_csv(f, header=f.tell() == 0, index=False)

        with open(self.savePath_snr + '/' + self.halodata.filename + '_noise_avg' + '.csv', 'w') as ff:
            noise_area_avg = self.mycanvas.area_snr_avg.area.flatten()
            noise_avg_shape = noise_area_avg.shape
            noise_avg_csv = pd.DataFrame.from_dict({'year': np.repeat(self.halodata.more_info['year'], noise_avg_shape),
                                                    'month': np.repeat(self.halodata.more_info['month'], noise_avg_shape),
                                                    'day': np.repeat(self.halodata.more_info['day'], noise_avg_shape),
                                                    'location': np.repeat(self.halodata.more_info['location'].decode('utf-8'), noise_avg_shape),
                                                    'systemID': np.repeat(self.halodata.more_info['systemID'], noise_avg_shape),
                                                    'noise': noise_area_avg - 1})
            noise_avg_csv.to_csv(ff, header=ff.tell() == 0, index=False)
        self.halodata.filter(variables=['beta_raw', 'v_raw', 'cross_signal', 'depo_raw'],
                             ref='co_signal',
                             threshold=self.mycanvas.area_snr.threshold)

        self.halodata.filter(variables=['cross_signal_averaged', 'depo_averaged_raw'],
                             ref='co_signal_averaged',
                             threshold=self.mycanvas.area_snr_avg.threshold)
        self.snr_status.setText('Noise filtered')

    def depo_timeprofile(self):
        self.mycanvas.depo_time(halo=self.halodata)

    def depo_timesave(self):
        i = self.mycanvas.depo_tp.i
        area_value = self.mycanvas.depo_tp.area[:, i]
        area_range = self.halodata.data['range'][self.mycanvas.depo_tp.maskrange]
        area_snr = self.halodata.data['co_signal'].transpose()[self.mycanvas.depo_tp.mask][:, i]
        area_vraw = self.halodata.data['v_raw'].transpose()[self.mycanvas.depo_tp.mask][:, i]
        area_betaraw = self.halodata.data['beta_raw'].transpose()[self.mycanvas.depo_tp.mask][:, i]
        area_cross = self.halodata.data['cross_signal'].transpose()[
            self.mycanvas.depo_tp.mask][:, i]

        # Calculate indice of maximum snr value
        max_i = np.argmax(area_snr)

        result = pd.DataFrame.from_dict([{
            'year': self.halodata.more_info['year'],
            'month': self.halodata.more_info['month'],
            'day': self.halodata.more_info['day'],
            'location': self.halodata.more_info['location'].decode('utf-8'),
            'systemID': self.halodata.more_info['systemID'],
            'time': self.halodata.data['time'][self.mycanvas.depo_tp.masktime][i],  # time as hour
            'range': area_range[max_i],  # range
            'depo': area_value[max_i],  # depo value
            'depo_1': area_value[max_i - 1],
            'co_signal': area_snr[max_i],  # snr
            'co_signal1': area_snr[max_i-1],
            'vraw': area_vraw[max_i],  # v_raw
            'beta_raw': area_betaraw[max_i],  # beta_raw
            'cross_signal': area_cross[max_i]  # cross_signal
        }])

        # sub folder for each date
        depo_sub_folder = self.savePath_depo + '/' + self.halodata.filename
        Path(depo_sub_folder).mkdir(parents=True, exist_ok=True)

        # Append to or create new csv file
        with open(depo_sub_folder + '/' + self.halodata.filename + '_depo.csv', 'a') as f:
            result.to_csv(f, header=f.tell() == 0, index=False)
        # save fig
        name_png = self.halodata.filename + '_' + \
            str(int(self.halodata.data['time']
                    [self.mycanvas.depo_tp.masktime][i]*1000)) + '.png'
        self.mycanvas.print_figure(depo_sub_folder + '/' + name_png,
                                   dpi=200)
        self.depo_save_status.setText(f'Saved {name_png}')

    def depo_wholeprofile(self):
        self.depo_w = self.mycanvas.depo_whole(halo=self.halodata)

    def depo_wholesave(self):
        n_values = self.mycanvas.depo_wp.time.shape[0]
        result = pd.DataFrame.from_dict({
            'year': np.repeat(self.halodata.more_info['year'], n_values),
            'month': np.repeat(self.halodata.more_info['month'], n_values),
            'day': np.repeat(self.halodata.more_info['day'], n_values),
            'location': np.repeat(self.halodata.more_info['location'].decode('utf-8'), n_values),
            'systemID': np.repeat(self.halodata.more_info['systemID'], n_values),
            'time': self.mycanvas.depo_wp.time,  # time as hour
            'range': self.mycanvas.depo_wp.range[self.mycanvas.depo_wp.max_snr_indx][0],  # range
            'depo': self.mycanvas.depo_wp.depo_max_snr,  # depo value
            'depo_1': self.mycanvas.depo_wp.depo_max_snr1,
            'co_signal': self.mycanvas.depo_wp.max_snr,  # snr
            'co_signal1': self.mycanvas.depo_wp.max_snr1,
            'vraw': np.take_along_axis(self.halodata.data['v_raw'].transpose()[self.mycanvas.depo_wp.mask],
                                       self.mycanvas.depo_wp.max_snr_indx,
                                       axis=0)[0],  # v_raw
            'beta_raw': np.take_along_axis(self.halodata.data['beta_raw'].transpose()[self.mycanvas.depo_wp.mask],
                                           self.mycanvas.depo_wp.max_snr_indx,
                                           axis=0)[0],  # beta_raw
            'cross_signal': np.take_along_axis(self.halodata.data['cross_signal'].transpose()[self.mycanvas.depo_wp.mask],
                                               self.mycanvas.depo_wp.max_snr_indx,
                                               axis=0)[0]  # cross_signal
        })

        # sub folder for each date
        depo_sub_folder = self.savePath_depo + '/' + self.halodata.filename
        Path(depo_sub_folder).mkdir(parents=True, exist_ok=True)

        # Append to or create new csv file
        with open(depo_sub_folder + '/' + self.halodata.filename + '_depo.csv', 'a') as f:
            result.to_csv(f, header=f.tell() == 0, index=False)
        # save fig
        name_png = self.halodata.filename + '_' + \
            f'{self.mycanvas.depo_wp.time.min()*100:.0f}' + '-' + \
            f'{self.mycanvas.depo_wp.time.max()*100:.0f}' + '.png'
        self.mycanvas.print_figure(depo_sub_folder + '/' + name_png,
                                   dpi=200)
        self.depo_save_status.setText(f'Saved {name_png}')

    def info(self):
        self.infoView = QTableView()
        self.infoContent = pandasModel(self.halodata.describe())
        self.infoView.setModel(self.infoContent)
        self.infoView.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.infoView.resizeColumnsToContents()
        self.infoView.show()


class myCanvas(FigureCanvas):

    def __init__(self):
        self.fig = Figure(figsize=(18, 9), dpi=100)
        FigureCanvas.__init__(self, self.fig)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def plotdata(self, halo, variables):
        self.fig.clear()
        axes = self.fig.subplots(nrows=4, ncols=2,
                                 sharey=True, sharex=True)
        ax = axes.flatten()
        for i, var in enumerate(variables):
            if 'beta_raw' in var:
                val = np.log10(halo.data.get(var)).transpose()
            else:
                val = halo.data.get(var).transpose()

            if 'average' in var:
                xvar = halo.data.get('time_averaged')
            else:
                xvar = halo.data.get('time')
            yvar = halo.data.get('range')
            if halo.cbar_lim.get(var) is None:
                vmin = None
                vmax = None
            else:
                vmin = halo.cbar_lim.get(var)[0]
                vmax = halo.cbar_lim.get(var)[1]
            axi = ax[i]
            p = axi.pcolormesh(xvar, yvar, val, cmap='jet',
                               vmin=vmin,
                               vmax=vmax)
            axi.set_xlim([0, 24])
            axi.yaxis.set_major_formatter(hd.m_km_ticks())
            axi.set_title(var)
            cbar = self.fig.colorbar(p, ax=axi, fraction=0.05)
            cbar.ax.set_ylabel(halo.units.get(var, None), rotation=90)
            cbar.ax.yaxis.set_label_position('left')
        self.fig.suptitle(halo.filename,
                          size=30,
                          weight='bold')
        lab_ax = self.fig.add_subplot(111, frameon=False)
        lab_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        lab_ax.set_xlabel('Time (h)', weight='bold')
        lab_ax.set_ylabel('Height (km)', weight='bold')
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        self.draw()
        return self.fig

    def snr_filter(self, halo, multiplier=3, multiplier_avg=3):
        self.fig.clear()
        spec = self.fig.add_gridspec(2, 2, width_ratios=[1, 1],
                                     height_ratios=[2, 1])
        ax1 = self.fig.add_subplot(spec[0, 0])
        ax2 = self.fig.add_subplot(spec[0, 1])
        ax3 = self.fig.add_subplot(spec[1, 0])
        ax4 = self.fig.add_subplot(spec[1, 1], sharex=ax3)

        p1 = ax1.pcolormesh(halo.data['time'],
                            halo.data['range'],
                            halo.data['co_signal'].transpose(),
                            cmap='jet', vmin=0.995, vmax=1.005)
        ax1.yaxis.set_major_formatter(hd.m_km_ticks())
        ax1.set_title('Choose background noise for co_signal')
        ax1.set_ylabel('Height (km)')
        ax1.set_xlabel('Time (h)')
        ax1.set_ylim(bottom=0)
        self.area_snr = hd.area_snr(halo.data['time'],
                                    halo.data['range'],
                                    halo.data['co_signal'].transpose(),
                                    ax1,
                                    ax3,
                                    type='kde',
                                    multiplier=multiplier,
                                    fig=self.fig)
        self.fig.colorbar(p1, ax=ax1)

        p2 = ax2.pcolormesh(halo.data['time_averaged'],
                            halo.data['range'],
                            halo.data['co_signal_averaged'].transpose(),
                            cmap='jet', vmin=0.995, vmax=1.005)
        ax2.yaxis.set_major_formatter(hd.m_km_ticks())
        ax2.set_title('Choose background noise for co_signal averaged')
        ax2.set_ylabel('Height (km)')
        ax2.set_xlabel('Time (h)')
        ax2.set_ylim(bottom=0)
        self.area_snr_avg = hd.area_snr(halo.data['time_averaged'],
                                        halo.data['range'],
                                        halo.data['co_signal_averaged'].transpose(),
                                        ax2,
                                        ax4,
                                        type='kde',
                                        multiplier=multiplier_avg,
                                        fig=self.fig)
        self.fig.colorbar(p2, ax=ax2)
        self.fig.suptitle(halo.filename, size=22,
                          weight='bold')
        self.fig.canvas.draw()

    def depo_time(self, halo):
        self.fig.clear()
        ax1 = self.fig.add_subplot(211)
        ax2 = self.fig.add_subplot(223)
        ax3 = self.fig.add_subplot(224, sharey=ax2)
        p = ax1.pcolormesh(halo.data['time'],
                           halo.data['range'],
                           np.log10(halo.data['beta_raw'].transpose()),
                           cmap='jet', vmin=halo.cbar_lim['beta_raw'][0],
                           vmax=halo.cbar_lim['beta_raw'][1])
        self.fig.colorbar(p, ax=ax1, fraction=0.05, pad=0.02)
        ax1.set_title('beta_raw')
        ax1.set_xlabel('Time (h)')
        ax1.set_xlim([0, 24])
        ax1.set_ylim(bottom=0)
        ax1.set_ylabel('Height (km)')
        ax1.yaxis.set_major_formatter(hd.m_km_ticks())
        self.fig.suptitle(halo.filename,
                          size=30,
                          weight='bold')
        self.fig.subplots_adjust(hspace=0.3, wspace=0.2)
        self.depo_tp = hd.area_timeprofile(halo.data['time'],
                                           halo.data['range'],
                                           halo.data['depo_raw'].transpose(),
                                           ax1,
                                           ax_snr=ax3,
                                           ax_depo=ax2,
                                           snr=halo.data['co_signal'].transpose(),
                                           fig=self.fig)

        self.draw()
        return self.fig

    def depo_whole(self, halo):
        self.fig.clear()
        ax1 = self.fig.add_subplot(311)
        ax2 = self.fig.add_subplot(323)
        ax3 = self.fig.add_subplot(325, sharex=ax2)
        ax4 = self.fig.add_subplot(324)
        ax5 = self.fig.add_subplot(326)
        p = ax1.pcolormesh(halo.data['time'],
                           halo.data['range'],
                           np.log10(halo.data['beta_raw'].transpose()),
                           cmap='jet', vmin=halo.cbar_lim['beta_raw'][0],
                           vmax=halo.cbar_lim['beta_raw'][1])
        self.fig.colorbar(p, ax=ax1, fraction=0.05, pad=0.02)
        ax1.set_title('beta_raw')
        ax1.set_xlabel('Time (h)')
        ax1.set_xlim([0, 24])
        ax1.set_ylim(bottom=0)
        ax1.set_ylabel('Height (km)')
        ax1.yaxis.set_major_formatter(hd.m_km_ticks())
        self.fig.suptitle(halo.filename,
                          size=30,
                          weight='bold')
        self.fig.subplots_adjust(hspace=0.3, wspace=0.2)
        self.depo_wp = hd.area_wholecloud(halo.data['time'],
                                          halo.data['range'],
                                          halo.data['depo_raw'].transpose(),
                                          ax1,
                                          ax_snr=ax3,
                                          ax_depo=ax2,
                                          ax_hist_depo=ax4,
                                          ax_hist_snr=ax5,
                                          snr=halo.data['co_signal'].transpose(),
                                          fig=self.fig)

        self.draw()
        return self.fig


class pandasModel(QtCore.QAbstractTableModel):
    """
    Class to populate a table view with a pandas dataframe
    """

    def __init__(self, data, parent=None):
        QtCore.QAbstractTableModel.__init__(self, parent)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    # def headerData(self, col, orientation, role):
    #     if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
    #         return self._data.columns[col]
    #     return None
    def headerData(self, rowcol, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return self._data.columns[rowcol]
        if orientation == QtCore.Qt.Vertical and role == QtCore.Qt.DisplayRole:
            return self._data.index[rowcol]
        return None


def main():
    app = QApplication(argv)
    window = Main()
    window.showFullScreen()  # Start at position full screen
    # window.showMaximized()  # Start position max screen
    app.exec_()


if __name__ == '__main__':
    main()
