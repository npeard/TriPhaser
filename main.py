#!/usr/bin/env python

import numpy as np
import sys
import Plot_1D
import Plot_2D
import PaperFigures
import timeit

import Speckle_2D

if __name__ == '__main__':
    np.random.seed(0x5EED)
    if len(sys.argv) == 1:
        # useDFT only controls whether the diffraction is calculated analytically or not

        #plot = Plot_1D.Plot_1D(num_atoms=4, num_pix=101, kmax = 3,
        # useDFT=False)
        #plot.plot_Object()
        #for n in range(3):
        #    plot.plot_Shot()
        #plot.plot_g2(num_shots=1000)
        #plot.plot_Intensities(num_shots=10000)
        #plot.plot_g3(num_shots=1000)
        #plot.plot_Intensity_Error()
        #plot.plot_cosPhi(num_shots=10000)
        #plot.plot_Closure(num_shots=1000, saveMem=False)
        #plot.plot_ClosurePhase(num_shots=1000, saveMem=False)
        #plot.plot_cosPhi_Error()
        #plot.plot_PhiSolver(num_shots=10000)
        #plot.plot_simple_PhiSolve(num_shots=10000)
        #plot.learnStructure(num_shots=10000)

        fluo = Speckle_2D.Fluorescence_2D(num_pix=20)
        start_time = timeit.default_timer()
        fluo.init_weights_4d()
        end_time = timeit.default_timer()
        res_orig = fluo.weights_4d
        execution_time = end_time - start_time
        print(execution_time)

        fluo = Speckle_2D.Fluorescence_2D(num_pix=20)
        start_time = timeit.default_timer()
        res_new = fluo.compute_weights_4d(num_pix=20)
        end_time = timeit.default_timer()
        execution_time = end_time - start_time
        print(execution_time)
        print(res_new.shape, res_orig.shape)
        print(np.array_equal(res_orig, res_new))

        # plot = Plot_2D.Plot_2D(num_pix=11, num_atoms=7, kmax=7, useCrystal =
        # False, useDFT=False)
        #plot = Plot_2D.Plot_2D(num_pix=11, num_atoms=3, kmax=2, useCrystal=False, useDFT=False)
        #plot.plot_Intensities(num_shots=1000)
        #plot.plot_Object()
        #for n in range(3):
        #     plot.plot_Shot()
        #plot.plot_Closure(num_shots=1000, saveMem=False)
        #plot.plot_ClosurePhase(num_shots=5000, saveMem=False)
        #plot.plot_cosPhi(num_shots=5000)
        #plot.plot_PhiSolver(num_shots=10000)
        #plot.plot_PhiSolver_manualSelect(num_shots=1000)
        #plot.learnStructure(num_shots=10000)

        #PaperFigures.Figure_1()
        #PaperFigures.Figure_5_Rows()
        #PaperFigures.Figure_1_Components()
        #PaperFigures.Figure_3()
        #PaperFigures.Figure_7()
        #PaperFigures.Figure_8()
        #PaperFigures.Figure_4()
        #PaperFigures.Figure_5()
        #PaperFigures.Figure_6_1D()
        #PaperFigures.Figure_2()
        #PaperFigures.Figure_PhaseRamp_HarmInv()
        #PaperFigures.Figure_CoarsePhase_Demo()

    else:
        print("Error: Unsupported number of command-line arguments")