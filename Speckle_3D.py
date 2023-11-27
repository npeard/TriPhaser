#!/usr/bin/env python

import numpy as np
import Xtal
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg

class Fluorescence_3D:
    def __init__(self, kmax=25, num_pix=51):
        print("Running in simulation mode...")
        self.Crystal = Xtal.Xtal(superLattice=(5,5,5))
        self.kmax = kmax
        self.num_pix = num_pix
        self.num_atoms = self.Xtal.get_number_atoms()
        self.coords = self.Crystal.get_positions()
        self.init_system()


    def init_system(self):
        print("Initializing system...")
        self.k_pix = np.mgrid[-self.kmax:self.kmax:1j * self.num_pix, -self.kmax:self.kmax:1j * self.num_pix, -self.kmax:self.kmax:1j * self.num_pix]

        # Define the coherent diffraction
        self.kr_product_x = np.multiply.outer(self.k_pix[0, :, :], self.coords[0, :])
        self.kr_product_y = np.multiply.outer(self.k_pix[1, :, :], self.coords[1, :])
        self.kr_product_z = np.multiply.outer(self.k_pix[2, :, :], self.coords[2, :])
        self.coh_ft = np.exp(-1j * (self.kr_product_x + self.kr_product_y + self.kr_product_z)).mean(2)
        self.coh_phase = np.angle(self.coh_ft)


    def display_3D_cohDiffract(self):
        app = QtGui.QApplication([])

        ## Create window with two ImageView widgets
        win = QtGui.QMainWindow()
        win.resize(1000, 1000)
        win.setWindowTitle('Triple Correlations')
        cw = QtGui.QWidget()
        win.setCentralWidget(cw)
        l = QtGui.QGridLayout()
        cw.setLayout(l)
        imv1 = pg.ImageView()
        imv2 = pg.ImageView()
        l.addWidget(imv1, 0, 0)
        l.addWidget(imv2, 1, 0)
        win.show()

        ## Set your 3D data to be displayed here
        data = np.abs(self.coh_ft)**2

        roi = pg.LineSegmentROI([[10, len(data) // 2], [len(data) - 10, len(data) // 2]], pen='r')
        imv1.addItem(roi)

        def update():
            global data, imv1, imv2
            d2 = roi.getArrayRegion(data, imv1.imageItem, axes=(1, 2))
            imv2.setImage(d2)

        roi.sigRegionChanged.connect(update)

        ## Display the data
        imv1.setImage(data)
        imv1.setHistogramRange(0, 2)
        imv1.setLevels(0, 2)

        update()
        import sys
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()
