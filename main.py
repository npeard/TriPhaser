#!/usr/bin/env python

import numpy as np
import sys
import Plot_1D
import Plot_2D
import PaperFigures
import timeit

import Speckle_1D
import Speckle_2D
import TriPhase_1D
import TriPhase_2D

if __name__ == '__main__':
    np.random.seed(0x5EED)
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

        # def test_get_g2():
        #     # Test case 1: g2 is already computed
        #     obj = Speckle_1D.Fluorescence_1D()
        #     obj.g2 = np.ones((10,))
        #     assert np.array_equal(obj.get_g2(), np.ones((10,)))
        #
        #     # Test case 2: g2 is not computed
        #     obj = Speckle_1D.Fluorescence_1D()
        #     obj.get_incoh_intens = lambda: np.ones((10,))
        #     obj.num_pix = 10
        #     assert np.array_equal(obj.get_g2(), np.ones((10,10)))
        #
        #     # Test case 3: num_shots = 1000
        #     obj = Speckle_1D.Fluorescence_1D()
        #     obj.get_incoh_intens = lambda: np.ones((10,))
        #     obj.num_pix = 10
        #     result = obj.get_g2(num_shots=1000)
        #     assert result.shape == (10,10)
        #     assert np.allclose(result, np.ones((10,10)))
        #     #assert np.allclose(result, np.zeros((10,10)))
        #
        # test_get_g2()
        fluo = Speckle_2D.Fluorescence_2D(num_pix=21)
        fluo.get_g3(num_shots=10)
        fluo.init_weights_4d()

        start_time = timeit.default_timer()

        #fluo = Speckle_2D.Fluorescence_2D(num_pix=101)
        g3_4d_orig = fluo.marginalize_g3(num_shots=10)

        end_time = timeit.default_timer()
        execution_time = end_time - start_time
        print(execution_time, "seconds")

        start_time = timeit.default_timer()

        g3_4d_new = fluo.compute_marginalized_g3(fluo.g3,
                                                 num_pix=21)
        g3_4d_new[fluo.weights_4d > 0] /= fluo.weights_4d[fluo.weights_4d > 0]

        end_time = timeit.default_timer()
        execution_time = end_time - start_time
        print(execution_time, "seconds")

        assert np.allclose(g3_4d_orig, g3_4d_new)
        print(np.allclose(g3_4d_orig, g3_4d_new))

    else:
        print("Error: Unsupported number of command-line arguments")