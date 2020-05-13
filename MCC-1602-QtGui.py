from CadicSoft_v2_ui import *
# from CadicSoft_TabsVersion import *
from SettingsWindow_ui import *

from _ctypes import POINTER, addressof, sizeof
from ctypes import c_ushort, cast, windll

import numpy as np
import pyqtgraph as pg
import time
import datetime
import os
import sys
from functools import partial
from copy import copy
from builtins import *  # @UnusedWildImport

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QObject, pyqtSlot, QRegExp, QThread, QMutex
from PyQt5.uic import loadUi

from mcculw import ul
from mcculw.enums import ScanOptions, FunctionType, Status
from examples.console import util
from examples.props.ai import AnalogInputProps
from mcculw.ul import ULError

import queue
from collections import deque
import scipy.io.wavfile
from soundfile import SoundFile
# import sounddevice as sd   UNCOMMENT PARA PLAY
# import samplerate
import resampy


def trap_exc_during_debug(*args):
    print(args)


sys.excepthook = trap_exc_during_debug

windll.shell32.SetCurrentProcessExplicitAppUserModelID('GIAS.GIASsoft.CADIC.1')


class Settings(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self):
        super(Settings, self).__init__()
        self.setupUi(self)
        self.setWindowTitle("Recorder Settings")
        self.setWindowIcon(QtGui.QIcon(
            'C:/Users/Camila/Desktop/Alex/Untref/Bioacústica/GIAS/MCC DAQ software python/GIAS_icon_3.png'))
        self.BrowseButton.clicked.connect(self.browseSlot)
        reg_ex = QRegExp("[^\\\/:*?<>|]+")
        input_validator = QtGui.QRegExpValidator(reg_ex, self.NamePrefixLabel)
        self.NamePrefixLabel.setValidator(input_validator)
        self.buttonBox.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(False)
        self.NamePrefixLabel.setToolTip(':*?<>|]+ characters not allowed')

        # validate settings
        self.FolderLabel.textChanged.connect(self.validate_settings)

    def validate_settings(self):
        if os.path.isdir(self.FolderLabel.text()) and os.path.exists(self.FolderLabel.text()):
            self.buttonBox.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(True)
        else:
            self.buttonBox.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(False)
            self.FolderLabel.setToolTip('Invalid Directory')

    def browseSlot(self):
        output_directory = QtWidgets.QFileDialog()
        self.fdir = str(output_directory.getExistingDirectory(
            self, 'Select Folder', 'C.\\'))
        if self.fdir != '':
            self.FolderLabel.setText('{}'.format(self.fdir))
            self.buttonBox.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(True)


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__()

        self.initUi()

    def initUi(self):
        pg.setConfigOption('background', 'w')  # before loading widget
        self.setupUi(self)
        self.setWindowTitle("GIAS Recorder")
        self.setWindowIcon(QtGui.QIcon(
            'C:/Users/Camila/Desktop/Alex/Untref/Bioacústica/GIAS/MCC DAQ software python/GIAS_icon.png'))
        self.StartButton.clicked.connect(self.start)
        self.StartButton.setIcon(QtGui.QIcon(
            'C:/Users/Camila/Desktop/Alex/Untref/Bioacústica/GIAS/MCC DAQ software python/Rec_button.png'))
        self.SettingsButton.clicked.connect(self.open_settings)
        self.StopButton.clicked.connect(self.stop)
        self.progressBar.setMaximum(32768)
        pg.setConfigOptions(antialias=True)
        self.rec_settings = Settings()
        self.BIPRange = 0
        self.srate = 500000
        self.CHUNKSZ = 1024  # int(self.srate/1000)
        self.plot_iter = 0
        self.setupThreads()
        self.open_settings()
        self.showMaximized()
        app.aboutToQuit.connect(self.forceWorkerQuit)

    def open_settings(self):
        self.rec_settings.exec_()
        self.BIPRange = self.rec_settings.Range.currentIndex()
        self.srate = int(self.rec_settings.SampleRate.currentText())*10**3
        self.CHUNKSZ = 1024   # int(self.srate/1000)
        self.set_plot_props()
        self.forceWorkerReset()

    def set_plot_props(self):
        # TODO: cehck for vispy module

        # Time Plot Properties
        self.win = self.graphicsView
        ############## test 1 PLotWidget ###############################################
        # self.win.plotItem.setRange(
        #     xRange=[0, self.CHUNKSZ], yRange=[-35000, 35000], padding=0,
        #     disableAutoRange=True)
        # self.win.setMouseEnabled(x=True, y=False)
        # self.win.showGrid(True, True, 0.5)
        # self.win.setLabel('bottom', 'Time', units='s')
        # self.win.setLabel('left', 'Amplitude', units='counts')
        ############## test 2 GraphicsLayoutWidget ####################################
        # self.win.clear()
        # self.data1 = np.zeros(2*self.CHUNKSZ)
        # self.plot_index = 0
        # self.p1 = self.win.addPlot(row=0, col=1)
        # self.p1.setDownsampling(mode='peak')
        # self.p1.setClipToView(True)
        # self.p1.setLimits(xMin=0, xMax=2*self.CHUNKSZ, yMin=-32768, yMax=32767)
        # self.p1.disableAutoRange()
        # self.p1.setXRange(0, 2*self.CHUNKSZ)
        # self.p1.setYRange(-32768, 32767)
        # self.curve1 = self.p1.plot(self.data1, pen='b')
        # self.win.addLabel('Time [s]', row=1, col=1)
        # self.win.addLabel('Amplitud [counts]', row=0, col=0, angle=-90)
        ############## test 3 GraphicsLayoutWidget ####################################
        self.win.clear()
        self.max_chunks = 10
        self.p1 = self.win.addPlot()
        self.p1.setDownsampling(mode='peak')
        self.p1.disableAutoRange()
        self.p1.setXRange(-self.CHUNKSZ, 0)
        self.p1.setYRange(-32768, 32767)
        self.curves = []
        self.ptr = 0
        self.data1 = np.empty((self.CHUNKSZ+1, 2))
        self.win.addLabel('Time [s]', row=1, col=1)
        # self.win.addLabel('Amplitud [counts]', row=0, col=0, angle=-90)
        ############## test 4 PlotWidget (arrayToQPath) ################################
        # self.win.clear()  # .plotItem
        # self.win.setMouseEnabled(x=True, y=False)
        # self.win.showGrid(True, True, 0.5)
        # self.win.setLabel('bottom', 'Time', units='s')
        # self.win.setLabel('left', 'Amplitude', units='counts')
        # self.ptr = 0
        # self.n = 5
        # self.win.setRange(
        #     xRange=[0, self.CHUNKSZ*self.n], yRange=[-35000, 35000], padding=0,
        #     disableAutoRange=True)
        # self.win.enableAutoRange(False, False)
        # self.x = np.arange(self.n*self.CHUNKSZ)
        # self.ydata = np.zeros((self.n*self.CHUNKSZ))
        # self.conn = np.ones((self.n*self.CHUNKSZ))
        # self.item = QtGui.QGraphicsPathItem()
        # self.item.setPen(pg.mkPen('b', width=2, style=QtCore.Qt.SolidLine))
        # self.win.addItem(self.item)

        #####################################################################################
        # Spectrogram Properties

        self.win2 = self.graphicsView_2
        self.win2.clear()
        self.win2.enableAutoRange(False, False)
        Xffts = 1024  # Number of ffts along X axis
        window_len = self.CHUNKSZ/self.srate  # Time window length
        spect_len = window_len * Xffts  # Spectrogram length

        self.img_array = np.zeros((Xffts, int(self.CHUNKSZ/2)+1))
        self.win2.plotItem.setRange(
            xRange=[0, spect_len-2*window_len], yRange=[0, self.srate/2+100], padding=0,
            disableAutoRange=True)
        self.win2.setMouseEnabled(x=True, y=True)
        self.win2.setLabel('left', 'Frequency', units='Hz')
        self.win2.setLabel('bottom', 'Time', units='s')
        # ax = self.win2.getAxis('bottom')
        # ax.setTicks([
        #     [],
        #     [], ])

        self.win2.setLimits(minXRange=0, maxXRange=spect_len,
                            minYRange=0, maxYRange=self.srate/2+100, xMin=0, xMax=spect_len,
                            yMin=0, yMax=self.srate/2+100)
        self.img = pg.ImageItem()
        self.img.setAutoDownsample(True)
        self.win2.addItem(self.img)

        # self.img_array = np.zeros((1000, int(self.CHUNKSZ/2)+1))
        self.img_index = 2
        # bipolar colormap

        pos = np.array([0., 1., 0.5, 0.25, 0.75])
        color = np.array([[0, 255, 255, 255], [255, 255, 0, 255], [0, 0, 0, 255],
                          (0, 0, 255, 255), (255, 0, 0, 255)], dtype=np.ubyte)
        cmap = pg.ColorMap(pos, color)
        lut = cmap.getLookupTable(0.0, 1.0, 256)

        self.img.setLookupTable(lut)
        self.img.setLevels([-50, 40])

        # freq = np.arange((256/2)+1)/(float(256)/self.srate)
        # yscale = 1.0/(self.img_array.shape[1]/freq[-1])
        # self.img.scale((1./self.srate)*256, yscale)
        freq = np.arange((self.CHUNKSZ/2)+1)/(float(self.CHUNKSZ)/self.srate)
        yscale = 1.0/(self.img_array.shape[1]/freq[-1])
        self.img.scale((1./self.srate)*self.CHUNKSZ, yscale)

        self.hann = np.hanning(self.CHUNKSZ)
        # self.empty_plot()  # UNCOMMENT DESPUES DE TESTEO TIME PLOT
################################### test 1 s####################################
        # # Add a histogram with which to control the gradient of the image
        # hist = pg.HistogramLUTItem()
        # # Link the histogram to the image
        # hist.setImageItem(self.img)
        # # If you don't add the histogram to the window, it stays invisible, but I find it useful.
        # self.win2.addItem(hist)
        # # Show the window
        # self.win2.show()
        #
        # # This gradient is roughly comparable to the gradient used by Matplotlib
        # # You can adjust it and then save it using hist.gradient.saveState()
        # hist.gradient.restoreState(
        #     {'mode': 'rgb',
        #      'ticks': [(0.5, (0, 182, 188, 255)),
        #                (1.0, (246, 111, 0, 255)),
        #                (0.0, (75, 0, 113, 255))]})

        # # Fit the min and max levels of the histogram to the data available
        # hist.setLevels(np.min(Sxx), np.max(Sxx))
################################################################################
    # Plot allocation

    def empty_plot(self):
        # self.timedata = np.zeros(self.CHUNKSZ)
        # self.x = np.arange(0, self.CHUNKSZ, 1)
        # self.set_plot_data(self.x, self.timedata)
        # self.data1 = np.zeros(5*self.CHUNKSZ)
        # self.curve1 = self.p1.plot(self.data1, pen='b')
        self.clipBar.setValue(0)

        self.image_index = 2

    def setupThreads(self):
        # Data Aqcuisition Thread
        self.daq_thread = QThread()
        self.daq_worker = DaqThread(self.BIPRange, self.srate)
        self.daq_worker.moveToThread(self.daq_thread)
        self.daq_thread.start()
        # conexiones Thread -> Main
        self.daq_worker.chunk_signal.connect(self.update)
        # conexiones Main -> Thread
        self.StartButton.clicked.connect(self.daq_worker.start_scan)
        # self.StopButton.clicked.connect(self.daq_worker.stop_daq_thread) NOT WORKING
        self.daq_worker.finished_signal.connect(self.forceWorkerReset)
        # self.daq_thread.finished.connect(self.worker.deleteLater)
        # # Playback Thread
        # self.play_thread = QThread()          UNCOMMENT PARA PLAY HASTA FINAL
        # self.play_worker = PlayThread(self.srate, self.CHUNKSZ)
        # self.play_worker.moveToThread(self.play_thread)
        # self.play_thread.start()
        # # Play thread signals
        # self.StartButton.clicked.connect(self.play_worker.open_stream)
        # self.StopButton.clicked.connect(self.play_worker.end_stream)
        # self.daq_worker.chunk_signal.connect(self.play_worker.get_signal)

    def stop(self):
        self.toggle_buttons()
        self.daq_worker.stop_daq_thread()
        self.empty_plot()  # UNCOMMENT DESPUES DE TESTEO PLOT
        # self.play_worker.end_stream() UNCOMMENT PARA PLAY

    def toggle_buttons(self):
        self.StartButton.setEnabled(True)
        self.StopButton.setEnabled(False)
        self.SettingsButton.setEnabled(True)
        self.filenameLabel.setText('Status: Idle')

    @pyqtSlot()
    def forceWorkerReset(self):
        if self.daq_thread.isRunning():
            # print('Quitting Thread')
            # self.daq_thread.quit

            # print('Terminating thread.')
            self.daq_thread.terminate()

            # print('Waiting for thread termination.')
            self.daq_thread.wait()

            self.toggle_buttons()

        # if self.play_thread.isRunning(): UNCOMMENT PARA PLAY
        #     # print('Quitting Thread')
        #     # self.daq_thread.quit
        #
        #     print('Terminating thread.')
        #     self.play_thread.terminate()
        #
        #     print('Waiting for thread termination.')
        #     self.play_thread.wait()

        # print('building new working object.')
        self.setupThreads()

    def forceWorkerQuit(self):
        if self.daq_thread.isRunning():
            self.daq_thread.terminate()
            self.daq_thread.wait()

        # if self.play_thread.isRunning(): UNCOMMENT PARA PLAY
        #     self.play_thread.terminate()
        #     self.play_thread.wait()

    def start(self):
        self.StopButton.setEnabled(True)
        self.StartButton.setEnabled(False)
        self.SettingsButton.setEnabled(False)
        # if self.rec_settings.FolderLabel.text() == '': # and monitor not selected
        #     QtGui.QMessageBox.critical(self, "No Destination Folder", "Please select settings")
        #     #algun comando para que no avance el programa
        self.filenameLabel.setText('Status: Recording to ' + self.rec_settings.FolderLabel.text() +
                                   '/' + self.rec_settings.NamePrefixLabel.text() + '...')
        self.startTime = pg.ptime.time()

    def set_plot_data(self, x, y):
        self.win.plot(x, y, pen=pg.mkPen('b', style=QtCore.Qt.SolidLine), clear=True)

    @QtCore.pyqtSlot(np.ndarray)
    def update(self, chunk):
        self.chunk1 = copy(chunk)
        # # time series rolling array
        # now = pg.ptime.time()
        # self.timedata = np.roll(self.timedata, -len(self.chunk1))
        # self.timedata[-len(self.chunk1):] = self.chunk1
        # self.set_plot_data(self.x, self.timedata)
        # print("Plot time: %0.4f sec" % (pg.ptime.time()-now))
############## test 1 #########################################################
        # self.data1[:len(self.chunk1)] = self.data1[-len(self.chunk1):]
        # self.data1[-len(self.chunk1):] = self.chunk1  # shift data in the array one sample left
        # self.curve1.setData(self.data1)
        # print("Plot time: %0.4f sec" % (pg.ptime.time()-now))
############## test 2 #########################################################
        # if self.plot_index == 20*self.CHUNKSZ:
        #     self.plot_index = 0
        # self.data1[self.plot_index:self.plot_index+self.CHUNKSZ] = self.chunk1
        # self.curve1.setData(self.data1)
        # self.plot_index += self.CHUNKSZ
        # print("Plot time: %0.4f sec" % (pg.ptime.time()-now))
############## test 3 #########################################################
        now = pg.ptime.time()
        for c in self.curves:
            c.setPos(-(self.CHUNKSZ), 0)

        curve = self.p1.plot(pen='b')
        self.curves.append(curve)
        if len(self.curves) > 2:  # > self.max_chunks
            c = self.curves.pop(0)
            self.p1.removeItem(c)
        curve.setData(x=np.arange(0, self.CHUNKSZ), y=self.chunk1)
        # print("Plot time: %0.4f sec" % (pg.ptime.time()-now))

############## test 4 (arrayToPyqtGraph) ######################################
        # now = pg.ptime.time()
        # if self.ptr == self.n*self.CHUNKSZ:
        #     self.ptr = 0
        #     self.ydata.fill(0)
        # self.ydata[self.ptr:self.ptr+self.CHUNKSZ] = self.chunk1
        # self.ptr += self.CHUNKSZ
        # # if self.ptr == int(self.n/2):
        # path = pg.arrayToQPath(self.x, self.ydata, self.conn)
        # self.item.setPath(path)
        # print("Plot time: %0.4f sec" % (pg.ptime.time()-now))
############### test 5 (resampling chunk test) ################################

######################################################################
        # Spectrogram
        # normalized, windowed frequencies in data chunk
        spec = np.fft.rfft(self.chunk1*self.hann) / self.CHUNKSZ  # add ,n=512/128/256
        # spec = np.fft.rfft(self.chunk1*self.hann, n=256) / self.CHUNKSZ  # add ,n=512/128/256

        # get magnitude
        psd = abs(spec)

        # # nivel del vúmetro
        chunk_abs = np.abs(self.chunk1)
        # avg_vumeter = np.average(chunk_abs)
        peak_val = np.max(chunk_abs)
        # self.progressBar.setValue(peak_val)}

        # Clip indicator
        if peak_val > 32760:
            self.clipBar.setValue(1)
        # convert to dB scale
        try:
            psd = 20 * np.log10(psd)
        except:
            psd[psd == 0] = np.amin(psd[psd != 0])
            psd = 20 * np.log10(psd)  # parche momentaneo

        # # roll down one and replace leading edge with new data
        # self.img_array = np.roll(self.img_array, -1, 0)
        # self.img_array[-1:] = psd

        # roll down one and replace leading edge with new data
        if self.img_index == len(self.img_array):
            self.img_index = 2

        self.img_array[self.img_index-2] = psd
        self.img_array[self.img_index-1:self.img_index+1] = 10*np.ones(len(psd))
        self.img_index += 1
        # self.plot_iter += 1
        # if self.plot_iter == 2:
        self.img.setImage(self.img_array, autoLevels=False)  # uncomment despues de time test
        # self.plot_iter = 0
##############################################################################


class DaqThread(QObject):
    chunk_signal = QtCore.pyqtSignal(np.ndarray)
    finished_signal = QtCore.pyqtSignal()

    def __init__(self, biprange, srate):
        super(DaqThread, self).__init__()
        # QObject.__init__(self)

        self.biprange = biprange

    # self.mutex = QMutex()
        self.rate = srate
        self.chunksize = 1024  # int(self.rate/1000)
        self.board_num = 0
        self.file_ls = []
        self.chunk_ls = []
        self.chunk_np = np.zeros(self.chunksize)
        ########
        # The size of the UL buffer to create, in seconds
        buffer_size_seconds = 15
        # The number of buffers to write
        num_buffers_to_write = 8

        # VER si eliminar o dejar para que confirme si recibe el device y
        # lo libere al final, esta bueno para no tener que abrir instacal
        self.use_device_detection = False  # Cambiar a True en version final
        if self.use_device_detection:
            ul.ignore_instacal()
            if not util.config_first_detected_device(self.board_num):
                QtGui.QMessageBox.information(
                    self, "Connect device", "Could not find device")  # check message box
                return

        ai_props = AnalogInputProps(self.board_num)
        # In case more channels are added in the future
        self.low_chan = 0
        self.high_chan = 0
        num_chans = self.high_chan - self.low_chan + 1

        # Create a circular buffer that can hold buffer_size_seconds worth of
        # data, or at least 10 points (this may need to be adjusted to prevent
        # a buffer overrun)
        points_per_channel = max(self.rate * buffer_size_seconds, 10)

        # Some hardware requires that the total_count is an integer multiple
        # of the packet size. For this case, calculate a points_per_channel
        # that is equal to or just above the points_per_channel selected
        # which matches that requirement.
        if ai_props.continuous_requires_packet_size_multiple:
            packet_size = ai_props.packet_size
            remainder = points_per_channel % packet_size
            if remainder != 0:
                points_per_channel += packet_size - remainder

        # In case more channels are added in the future
        self.ul_buffer_count = points_per_channel * num_chans

        # Pick range from settings Combobox
        self.ai_range = ai_props.available_ranges[self.biprange]

        # Write the UL buffer to the file num_buffers_to_write times
        self.points_to_write = self.ul_buffer_count * num_buffers_to_write

        # # When handling the buffer, we will read 1/10 of the buffer at a time
        self.write_chunk_size = self.chunksize  # int(self.ul_buffer_count / 10)

        self.scan_options = (ScanOptions.BACKGROUND | ScanOptions.CONTINUOUS)

        self.memhandle = ul.win_buf_alloc(self.ul_buffer_count)

        # Allocate an array of doubles temporary storage of the data
        self.write_chunk_array = (c_ushort * self.write_chunk_size)()

        # Check if the buffer was successfully allocated
        if not self.memhandle:
            QtGui.QMessageBox.critical(self, "No Buffer", "Failed to allocate memory.")
            print("No Buffer")
            ul.stop_background(self.board_num, FunctionType.AIFUNCTION)
            return

    @pyqtSlot()
    def start_scan(self):
        # Set filename
        self.file_name = window.rec_settings.FolderLabel.text() + '/' + window.rec_settings.NamePrefixLabel.text() + \
            datetime.datetime.now().strftime("_%Y_%m_%d_%H%M%S") + \
            '.wav'

        try:
            # Start the scan
            ul.a_in_scan(
                self.board_num, self.low_chan, self.high_chan, self.ul_buffer_count,
                self.rate, self.ai_range, self.memhandle, self.scan_options)

            self.status = Status.IDLE
            # Wait for the scan to start fully
            while(self.status == Status.IDLE):
                self.status, _, _ = ul.get_status(
                    self.board_num, FunctionType.AIFUNCTION)

            # Create a file for storing the data
            # PYSOUNDFILE MODULE
            temp_file = SoundFile(self.file_name, 'w+', self.rate, 1, 'PCM_16')
            # with SoundFile(self.file_name, 'w', self.rate, 1, 'PCM_16') as f:
            #     print('abro', self.file_name)
            # WAVE MODULE
            # with wave.open('wavemod' + self.file_name, 'w') as f:
            #     f.setnchannels(1)
            #     f.setsampwidth(2)
            #     f.setframerate(self.rate)

            # Start the write loop
            prev_count = 0
            prev_index = 0
            write_ch_num = self.low_chan

            while self.status != Status.IDLE:
                # Get the latest counts
                self.status, curr_count, _ = ul.get_status(
                    self.board_num, FunctionType.AIFUNCTION)

                new_data_count = curr_count - prev_count

                # Check for a buffer overrun before copying the data, so
                # that no attempts are made to copy more than a full buffer
                # of data
                if new_data_count > self.ul_buffer_count:
                    # Print an error and stop writing
                    # QtGui.QMessageBox.information(self, "Error", "A buffer overrun occurred")
                    ul.stop_background(self.board_num, FunctionType.AIFUNCTION)
                    print("A buffer overrun occurred")  # cambiar por critical message
                    break  # VER COMO REEMPLAZAR

                # Check if a chunk is available
                if new_data_count > self.write_chunk_size:
                    self.wrote_chunk = True
                    # Copy the current data to a new array

                    # Check if the data wraps around the end of the UL
                    # buffer. Multiple copy operations will be required.
                    if prev_index + self.write_chunk_size > self.ul_buffer_count - 1:
                        first_chunk_size = self.ul_buffer_count - prev_index
                        second_chunk_size = (
                            self.write_chunk_size - first_chunk_size)

                        # Copy the first chunk of data to the write_chunk_array
                        ul.win_buf_to_array(
                            self.memhandle, self.write_chunk_array, prev_index,
                            first_chunk_size)

                        # Create a pointer to the location in
                        # write_chunk_array where we want to copy the
                        # remaining data
                        second_chunk_pointer = cast(
                            addressof(self.write_chunk_array) + first_chunk_size
                            * sizeof(c_ushort), POINTER(c_ushort))

                        # Copy the second chunk of data to the
                        # write_chunk_array
                        ul.win_buf_to_array(
                            self.memhandle, second_chunk_pointer,
                            0, second_chunk_size)
                    else:
                        # Copy the data to the write_chunk_array
                        ul.win_buf_to_array(
                            self.memhandle, self.write_chunk_array, prev_index,
                            self.write_chunk_size)

                    # Check for a buffer overrun just after copying the data
                    # from the UL buffer. This will ensure that the data was
                    # not overwritten in the UL buffer before the copy was
                    # completed. This should be done before writing to the
                    # file, so that corrupt data does not end up in it.
                    self.status, curr_count, _ = ul.get_status(
                        self.board_num, FunctionType.AIFUNCTION)
                    # Opcion 1: original ( valores altos )
                    if curr_count - prev_count > self.ul_buffer_count:
                        # Print an error and stop writing
                        ul.stop_background(self.board_num, FunctionType.AIFUNCTION)
                        print("BUFFER OVERRUN")
                        QtGui.QMessageBox.critical(self, "Warning", "A buffer overrun occurred")
                        break
                        # VER COMO HACER PARA EVITAR QUE CIERRE EL PROGRAMA:

                    for i in range(self.write_chunk_size):

                        # opcion 1
                        self.chunk_ls.append(self.write_chunk_array[i]-32768)

                        # # opcion 2
                        # write_data = struct.pack("<h", self.write_chunk_array[i])
                        # f.writeframesraw(write_data)

                        # # opcion 3
                        # f.writeframes(str(self.write_chunk_array[i]))
                        # # Ver si va str o binary o que formato ?????
                        # Write_chunk_array[i] es un int
                        # f.writeframes(b''.join(self.write_chunk_array))

                    # opcion 4
                    self.chunk_np = np.asarray(self.chunk_ls, dtype=np.int16)
                    # resampled_chunk = samplerate.resample(self.chunk_np, 44100. /
                    #                                       float(self.rate), 'sinc_best')
                    # resampled_chunk = resampy.resample(self.chunk_np, self.rate, 44100)

                    temp_file.write(self.chunk_np)
                    # self.chunk_signal.emit(self.chunk_ls)
                    # self.file_ls.extend(self.chunk_ls)
                    self.chunk_ls = []

                    # # opcion 2
                    # for i in range(self.write_chunk_size):
                    #     write_data = struct.pack("<h", self.write_chunk_array[i])
                    #     f.writeframesraw(write_data)

                    # # opcion 3
                    # for i in range(self.write_chunk_size):
                    #     f.writeframes(str(self.write_chunk_array[i]))
                    #     # Ver si va str o binary o que formato ?????
                    #     f.writeframes(b''.join(self.write_chunk_array))

                    # if len(self.file_ls) % self.ul_buffer_count == 0:
                    #     print("wrote buffer")

                else:
                    self.wrote_chunk = False

                if self.wrote_chunk:
                    self.chunk_signal.emit(self.chunk_np)
                    # Increment prev_count by the chunk size
                    prev_count += self.write_chunk_size
                    # Increment prev_index by the chunk size
                    prev_index += self.write_chunk_size
                    # Wrap prev_index to the size of the UL buffer
                    prev_index %= self.ul_buffer_count

                    if prev_count % self.points_to_write == 0:
                        # self.file_signal.emit(self.file_np)
                        # self.write_wav_file(self.file_ls
                        temp_file.close()
                        self.file_name = window.rec_settings.FolderLabel.text() + '/' + window.rec_settings.NamePrefixLabel.text() + \
                            datetime.datetime.now().strftime("_%Y_%m_%d_%H%M%S") + \
                            '.wav'
                        temp_file = SoundFile(self.file_name, 'w', self.rate, 1, 'PCM_16')
                else:
                    # Wait a short amount of time for more data to be
                    # acquired.
                    time.sleep(0.1)
        except ULError as e:
            print('except')
            # QtGui.QMessageBox.critical(window, 'Error', 'Please restart program')
            self.print_ul_error(e)  # VER FUNCION Y ADAPATAR A PYQT
        finally:
            # Free the buffer in a finally block to prevent errors from causing
            # a memory leak.
            temp_file.close()
            ul.stop_background(self.board_num, FunctionType.AIFUNCTION)
            ul.win_buf_free(self.memhandle)
            self.finished_signal.emit()
            # if self.use_device_detection:
            #     ul.release_daq_device(self.board_num)

    def print_ul_error(self, ul_error):
        # message = (
        #     "A UL Error occurred.\n\n Error Code: " + str(ul_error.errorcode)
        #     + "\nMessage: " + ul_error.message)
        print(str(ul_error.errorcode))
        # QtGui.QMessageBox.critical(window, str(ul_error.errorcode), ul_error.message)

    # @pyqtSlot() NOT WORKING
    def stop_daq_thread(self):
        print('Stop daq thread')
        self.status = Status.IDLE
        ul.stop_background(self.board_num, FunctionType.AIFUNCTION)

        # if self.use_device_detection:
        #     ul.release_daq_device(self.board_num)

    # def __del__(self):
    #     self.wait()


# class PlayThread(QObject):
#     def __init__(self, srate, chunksize):
#         super(PlayThread, self).__init__()

####################### Opcion 1 ########################
    #     self.srate = srate
    #     self.playback_buffer = deque()  # No me deja iterar sobre numpy indexes, trabajar con listas
    #     self.data = []
    #
    # @QtCore.pyqtSlot(np.ndarray)
    # def get_signal(self, chunk):
    #     print('entro en get_signal')
    #     chunk2 = copy(chunk)
    #     resampled_chunk = resampy.resample(chunk2, self.srate, 44100)
    #     self.playback_buffer.extend(resampled_chunk)
    #
    # def callback(self, outdata, frames, time, status):
    #     if status:
    #         print("Playback Error: {}".format(status))
    #     print('callback', 'frames', frames)
    #     print('PB antes', len(self.playback_buffer))
    #     try:
    #         if len(self.playback_buffer) < 5*frames:
    #             print('check 1')
    #             outdata[len(playback_buffer):, 0].fill(0)
    #         else:
    #             for i in range(frames):
    #                 self.data.append(self.playback_buffer.popleft())
    #             self.data_np = np.asarray(self.data, dtype=np.int16)
    #             self.data = []
    #             if len(self.data_np) < len(outdata):  # no debería pasar nunca???
    #                 print('check 4')
    #                 raise sd.CallbackStop()
    #             else:
    #                 print('check 5')
    #                 outdata[:, 0] = self.data_np
    #     except:
    #         print('callback end')
    #         self.end_stream()
    # #
    #
    # def open_stream(self):
    #     print('open stream')
    #     self.stream = sd.OutputStream(channels=1, blocksize=4096, callback=self.callback)
    #     self.stream.start()
    #             # with sd.OutputStream(channels=1, callback=self.callback) as s:  pareceria bloquear el thread entero y get signal no recibe
    #             #     print('en algun logar de stream')
    #         #         while s.active:
    #         #             time.sleep(0.1)
    #         # #
    #
    # def end_stream(self):
    #     print('end stream')
    #     raise sd.CallbackStop()
    #     self.stream.abort()
    #     self.stream.close()

# Opcion 2 ########################## mejor opcion por ahora UNCOMMENT PARA PLAY
# PUESTO PARA QUE LEA DIRECTO ARRAY (90SAMPLES), VER COMO ESTIRAR A 2048?
    #     self.srate = srate
    #     self.chunksize = chunksize
    #     self.playback_buffer = deque()
    #     self.data = []
    #     self.resampled_rate = 44100
    #     self.res_chunk_size = round((self.resampled_rate * self.chunksize) / self.srate)
    #
    # @QtCore.pyqtSlot(np.ndarray)
    # def get_signal(self, chunk):
    #     print('entro en get_signal')
    #     chunk2 = copy(chunk)
    #     resampled_chunk = resampy.resample(chunk2, self.srate, self.resampled_rate)
    #     self.playback_buffer.append(resampled_chunk)
    #
    # def callback(self, outdata, frames, time, status):
    #     if status:
    #         print("Playback Error: {}".format(status))
    #     print('callback')
    #     try:
    #         if len(self.playback_buffer) == 0:
    #             outdata[:, 0].fill(0)
    #         else:
    #             data = self.playback_buffer.popleft()
    #             print('frames', frames)
    #             outdata[:, 0] = data
    #     except:
    #         print('callback emd')
    #         self.end_stream()
    #
    # def open_stream(self):
    #     print('open stream')
    #     self.stream = sd.OutputStream(
    #         channels=1, blocksize=self.res_chunk_size, callback=self.callback)
    #     self.stream.start()
    #                 # !!! seems to freeze rest of the thread
    #                 # with sd.OutputStream(channels=1, callback=self.callback) as s: # blocksize=cambia frames???, latency=float permite sincrinzar con espectrograma
    #                 #     while s.active:
    #                 #         time.sleep(0.1)
    #
    # def end_stream(self):
    #     print('end sream')
    #     raise sd.CallbackStop()
    #     self.stream.abort()
    #     self.stream.close()
####################### Opcion 3 soundevice documentation ###########################################################

    # # Sounddevice using queue
    # blocksize = 2048
    # buffersize = 20
    # q = queue.Queue(maxsize=buffersize)
    #
    # def callback(outdata, frames, time, status):
    #     assert frames == blocksize
    #     if status.output_underflow:
    #         print('Output underflow: increase blocksize?', file=sys.stderr)
    #     raise sd.CallbackAbort
    #     assert not status
    #     try:
    #         data = q.get_nowait()
    #     except queue.Empty:
    #         print('Buffer is empty: increase buffersize?', file=sys.stderr)
    #         raise sd.CallbackAbort
    #     if len(data) < len(outdata):
    #         outdata[:len(data)] = data
    #         outdata[len(data):] = b'\x00' * (len(outdata) - len(data))
    #         raise sd.CallbackStop
    #     else:
    #         outdata[:] = data
    #
    # @pyqtSlot(np.ndarray)
    # def add_chunk(self, chunk):
    #     print('received chunk', len(chunk))
    #     try:
    #         data = chunk
    #         q.put_nowait(data)
    #
    #         outstream = sd.OutputStream(samplerate=44100, blocksize=blocksize,
    #                                     channels=1, dtype=np.int16, callback=callback)
    #
    #     except Exception as e:
    #         print(type(e).__name__ + ': ' + str(e))

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

    #########################################

    # Si tengo problemas con numeros muy grandes:
    # 1) cuando escribo file pongo prev_count = 0; en los condicionales
    #    multiplico prev_count por el "num_file_index" para llevarlo a su
    #    valor real
    # 2) cuando escribo file pongo prev_count = 0; en los condicionales
    #    divido curr_count por el "num_file_index" para que sea comparable
    #    con prev_count y se manejen valores bajos

    # def quit(self):
    #     choice = QtWidgets.QMessageBox.question(self, 'GIAS',
    #                                             "Are you sure you want to quit?",
    #                                             QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
    #     if choice == QtWidgets.QMessageBox.Yes:
    #         self.get_thread.close()
    #         self.get_thread.quit()
    #         sys.exit()
    #     else:
    #         pass


# ui to py gui: pyuic5 input.ui -o output.py
