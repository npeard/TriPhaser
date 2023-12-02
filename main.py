#!/usr/bin/env python

import numpy as np
import sys
import Plot_1D
import Plot_2D
import PaperFigures
import timeit

import Speckle_1D
import TriPhase_1D
import TriPhase_2D

if __name__ == '__main__':
    np.random.seed(0x5EED+1)
    if len(sys.argv) == 1:
        """Run functions in this scratch area. 
        """
        #plot = Plot_1D.Plot_1D(num_atoms=5, num_pix=101, kmax=10)
        #plot.plot_Object()
        #for n in range(3):
        #    plot.plot_Shot()
        #plot.plot_g2(num_shots=1000)
        #plot.plot_Intensities(num_shots=10000)
        #plot.plot_g3(num_shots=1000)
        #plot.plot_Intensity_Error()
        #plot.plot_cosPhi(num_shots=100000)
        #plot.plot_Closure(num_shots=1000)
        #plot.plot_ClosurePhase(num_shots=1000)
        #plot.plot_PhiSolver(num_shots=10000)
        #plot.plot_simple_PhiSolve(num_shots=1000)
        #plot.learnStructure(num_shots=10000)

        #plot = Plot_2D.Plot_2D(num_pix=11, num_atoms=7, kmax=2)
        #plot.plot_Object()
        #for n in range(3):
        #     plot.plot_Shot()
        #plot.plot_Intensities(num_shots=1000)
        #plot.plot_Closure(num_shots=1000)
        #plot.plot_ClosurePhase(num_shots=1000)
        #plot.plot_cosPhi(num_shots=1000)
        #plot.plot_PhiSolver(num_shots=20000)
        #plot.plot_PhiSolver_manualSelect(num_shots=1000)
        #plot.learnStructure(num_shots=10000)

        # TriPhase_1D.generate_training_data(num_data=100,
        #                                    file="/Users/nolanpeard/Desktop/Test1D-kmax3.h5")

        #TriPhase_2D.generate_training_data(num_data=100,
        #
    #                                   file="/Users/nolanpeard/Desktop/Test2D-kmax2.h5")

        #PaperFigures.Figure_1()
        #PaperFigures.Figure_Intro_Components()
        #PaperFigures.Figure_5_Rows()
        #PaperFigures.Figure_1_Components()
        #PaperFigures.Figure_3()
        #PaperFigures.Figure_7()
        #PaperFigures.Figure_8()
        #PaperFigures.Figure_4()
        #PaperFigures.Figure_5()
        #PaperFigures.Figure_6_1D()
        PaperFigures.Figure_2()
        #PaperFigures.Figure_PhaseRamp_HarmInv()
        #PaperFigures.Figure_CoarsePhase_Demo()

    else:
        print("Error: Unsupported number of command-line arguments")