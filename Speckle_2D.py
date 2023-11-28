#!/usr/bin/env python

import numpy as np
import scipy.signal
import Xtal
from scipy import optimize
from itertools import permutations
from numba import jit


class Fluorescence_2D:
    def __init__(self, kmax=5, num_pix=51, num_atoms=3, useCrystal=False,
                 x=None):
        """Form a complex number.

        Keyword arguments:
        real -- the real part (default 0.0)
        imag -- the imaginary part (default 0.0)
        """
        print("Running in simulation mode...")
        self.kmax = kmax
        self.num_pix = num_pix
        self.useCrystal = useCrystal
        self.x = x # User-supplied coordinates
        if self.useCrystal is True:
            print("Using crystal structure obtained from Xtal class...")
            self.Crystal = Xtal.Xtal()
            self.num_atoms = self.Crystal.get_number_atoms()
            self.init_system()
        else:
            print("Using randomized points as structure...")
            self.num_atoms = num_atoms
            self.init_system()


    def init_system(self):
        """Form a complex number.

        Keyword arguments:
        real -- the real part (default 0.0)
        imag -- the imaginary part (default 0.0)
        """
        print("Initializing system...")
        self.k_pix = np.mgrid[-self.kmax:self.kmax:1j * self.num_pix, -self.kmax:self.kmax:1j * self.num_pix]
        self.x_pix = np.mgrid[-1:1:1j * self.num_pix, -1:1:1j * self.num_pix]
        self.x_double_pix = np.mgrid[-1:1:1j * (2*self.num_pix-1), -1:1:1j * (2*self.num_pix-1)]
        self.weights_2d = np.correlate(np.ones(self.num_pix), np.ones(self.num_pix), mode='full')
        self.weights_2d = np.multiply.outer(self.weights_2d, self.weights_2d)
        self.q_pix = np.mgrid[-2 * self.kmax:2 * self.kmax:(2*self.num_pix-1) * 1j,
                     -2 * self.kmax:2 * self.kmax:(2*self.num_pix-1) * 1j]
        self.g2 = None
        self.g3 = None
        self.g2_2d = None
        self.g3_4d = None
        self.weights_4d = None

        if self.useCrystal is True:
            self.crystal_coords()
        else:
            self.randomize_coords()


    def init_weights_4d(self):
        """Form a complex number.

        Keyword arguments:
        real -- the real part (default 0.0)
        imag -- the imaginary part (default 0.0)
        """
        # This is a huge matrix, do not initialize it unless you need it!
        k1x, k2x, k3x, k1y, k2y, k3y = np.indices(6 * (self.num_pix,))
        # Check the ordering of elements here
        q1x = k1x - k2x
        q1y = k1y - k2y
        q2x = k2x - k3x
        q2y = k2y - k3y
        q1x -= q1x.min()
        q1y -= q1y.min()
        q2x -= q2x.min()
        q2y -= q2y.min()
        self.weights_4d = np.zeros(4 * (len(self.weights_2d),))
        np.add.at(self.weights_4d, tuple([q1x,q1y,q2x,q2y]), 1)

    @jit(nopython=True)
    def init_weights_4d_explicit(self):
        """Calculate the 4D weights using explicit for-loops.

        Keyword arguments:
        None
        """
        self.weights_4d = np.zeros(4 * (len(self.weights_2d),))

        for k1x in range(self.num_pix):
            for k2x in range(self.num_pix):
                for k3x in range(self.num_pix):
                    for k1y in range(self.num_pix):
                        for k2y in range(self.num_pix):
                            for k3y in range(self.num_pix):
                                q1x = k1x - k2x + self.num_pix - 1
                                q1y = k1y - k2y + self.num_pix - 1
                                q2x = k2x - k3x + self.num_pix - 1
                                q2y = k2y - k3y + self.num_pix - 1
                                self.weights_4d[q1x, q1y, q2x, q2y] += 1

    def randomize_coords(self):
        """Form a complex number.

        Keyword arguments:
        real -- the real part (default 0.0)
        imag -- the imaginary part (default 0.0)
        """
        self.coords = np.random.random((2, self.num_atoms)) * 2 - 1

        if self.x is not None:
            self.coords = self.x

        # Define the object for plotting
        # This is a real space object, but we place it in a discrete real space with the same number of bins as k-space for DFT
        self.object = np.zeros_like(self.x_pix[0,:,:])
        self.object[np.digitize(self.coords[0,:], self.x_pix[0,:,0]), np.digitize(self.coords[1,:], self.x_pix[1,0,:])] = 1/self.num_atoms
        self.object_double = np.zeros_like(self.x_double_pix[0,:,:])
        self.object_double[np.digitize(self.coords[0,:], self.x_double_pix[0,:,0]), np.digitize(self.coords[1,:], self.x_double_pix[1,0,:])] = 1/self.num_atoms
        # object_double is NOT the same object with double sampling, it is slightly different in the binning

        # Define the coherent diffraction
        self.kr_product_x = np.multiply.outer(self.k_pix[0, :, :], self.coords[0, :])
        self.kr_product_y = np.multiply.outer(self.k_pix[1, :, :], self.coords[1, :])
        self.qr_product_x = np.multiply.outer(self.q_pix[0, :, :], self.coords[0, :])
        self.qr_product_y = np.multiply.outer(self.q_pix[1, :, :], self.coords[1, :])
        self.coh_ft = np.exp(-1j * (self.kr_product_x + self.kr_product_y + 0)*2*np.pi).mean(2)
        self.coh_phase = np.angle(self.coh_ft)
        self.coh_ft_double = np.exp(-1j * (self.qr_product_x + self.qr_product_y + 0)*2*np.pi).mean(2)
        self.coh_phase_double = np.angle(self.coh_ft_double)

    def crystal_coords(self):
        """Form a complex number.

        Keyword arguments:
        real -- the real part (default 0.0)
        imag -- the imaginary part (default 0.0)
        """
        self.coords = self.Crystal.get_positions().T
        print(self.coords)

        # Define the object for plotting
        # This is a real space object, but we place it in a discrete real space with the same number of bins as k-space for DFT
        self.object = np.zeros_like(self.x_pix[0, :, :])
        self.object[np.digitize(self.coords[0, :], self.x_pix[0, :, 0]), np.digitize(self.coords[1, :], self.x_pix[1, 0,
                                                                                                        :])] = 1 / self.num_atoms
        self.object_double = np.zeros_like(self.x_double_pix[0, :, :])
        self.object_double[np.digitize(self.coords[0, :], self.x_double_pix[0, :, 0]), np.digitize(self.coords[1, :],
                                                                                                   self.x_double_pix[1,
                                                                                                   0,
                                                                                                   :])] = 1 / self.num_atoms
        # object_double is NOT the same object with double sampling, it is slightly different in the binning

        # Define the coherent diffraction
        #self.kz = 0.000001
        self.kr_product_x = np.multiply.outer(self.k_pix[0, :, :], self.coords[0, :])
        self.kr_product_y = np.multiply.outer(self.k_pix[1, :, :], self.coords[1, :])
        #self.kr_product_z = np.multiply.outer(self.kz * np.ones_like(self.k_pix[1, :, :]), self.coords[2, :])
        self.coh_ft = np.exp(-1j * (self.kr_product_x + self.kr_product_y + 0) * 2 * np.pi).mean(2)
        self.coh_phase = np.angle(self.coh_ft)
        self.qr_product_x = np.multiply.outer(self.q_pix[0, :, :], self.coords[0, :])
        self.qr_product_y = np.multiply.outer(self.q_pix[1, :, :], self.coords[1, :])
        self.coh_ft_double = np.exp(-1j * (self.qr_product_x + self.qr_product_y + 0) * 2 * np.pi).mean(2)
        self.coh_phase_double = np.angle(self.coh_ft_double)


    def get_incoh_intens(self):
        """Form a complex number.

        Keyword arguments:
        real -- the real part (default 0.0)
        imag -- the imaginary part (default 0.0)
        """
        incoh = np.abs(np.exp(-1j * ((self.kr_product_x + self.kr_product_y
                                       + np.random.random((self.num_atoms)))
                                      * 2. * np.pi)).mean(2))**2
        return incoh


    def get_g2(self, num_shots=1000):
        """Form a complex number.

        Keyword arguments:
        real -- the real part (default 0.0)
        imag -- the imaginary part (default 0.0)
        """
        if self.g2 is not None:
            return self.g2

        print("Performing second-order correlation using outer product...")
        ave_intens = np.zeros(2 * (self.num_pix,))
        self.g2 = np.zeros(4 * (self.num_pix,))

        for i in range(num_shots):
            print("Correlating frame ", i)
            incoh = self.get_incoh_intens()
            self.g2 += np.multiply.outer(incoh, incoh)
            ave_intens += incoh

        return self.g2



    def g2_fft(self, num_shots=1000):
        """Form a complex number.

        Keyword arguments:
        real -- the real part (default 0.0)
        imag -- the imaginary part (default 0.0)
        """
        # Uses FFT convolution to obtain the marginalized g2. See cross-correlation theorem.
        if self.g2_2d is not None:
            return self.g2_2d

        print("Performing second-order correlation using FFT...")
        self.g2_2d = np.zeros_like(self.q_pix[0, :, :])

        for i in range(num_shots):
            this_frame = self.get_incoh_intens()
            #this_frame /= np.max(this_frame)
            add = scipy.signal.fftconvolve(this_frame, this_frame[::-1, ::-1])
            self.g2_2d += add

        self.g2_2d /= self.weights_2d * num_shots / self.num_atoms**2
        #g2_2d /= np.mean(g2_2d)
        # Remove negative values
        self.g2_2d[(self.g2_2d - 1 + 1 / self.num_atoms) < 0] = 1 - 1 / self.num_atoms
        print("Finished correlation...")
        return self.g2_2d


    def marginalize_g2(self, num_shots=1000):
        """Form a complex number.

        Keyword arguments:
        real -- the real part (default 0.0)
        imag -- the imaginary part (default 0.0)
        """
        if self.g2_2d is not None:
            return self.g2_2d

        if self.g2 is None:
            self.g2 = self.get_g2(num_shots=num_shots)
        q_2d = np.subtract.outer(np.arange(self.num_pix), np.arange(self.num_pix))
        q_2d -= q_2d.min()

        # Better, but would be nice if the indexing worked directly for 4D matrices, or we need to Cythonize this
        self.g2_2d = np.zeros_like(self.weights_2d)
        for k1x in range(self.num_pix):
            for k2x in range(self.num_pix):
                np.add.at(self.g2_2d[self.num_pix - 1 + k1x - k2x, :], q_2d, self.g2[k1x, :, k2x, :])

        # Explicit quadruple for loops, very slow
        # for k1x in range(self.num_pix):
        #     for k2x in range(self.num_pix):
        #         for k1y in range(self.num_pix):
        #             for k2y in range(self.num_pix):
        #                 g2_2d[self.num_pix - 1 + k1x - k2x, self.num_pix - 1 + k1y - k2y] += self.g2[k1x,k1y,k2x,k2y]
        self.g2_2d[self.weights_2d > 0] /= self.weights_2d[self.weights_2d > 0]

        return self.g2_2d


    def get_g3(self, num_shots=1000):
        """Form a complex number.

        Keyword arguments:
        real -- the real part (default 0.0)
        imag -- the imaginary part (default 0.0)
        """
        if self.g3 is None:
            ave_intens = np.zeros(2 * (self.num_pix,))
            self.g3 = np.zeros(6 * (self.num_pix,))
            for i in range(num_shots):
                incoh = self.get_incoh_intens()
                self.g3 += np.multiply.outer(np.multiply.outer(incoh, incoh), incoh)
                ave_intens += incoh
            self.g3 *= num_shots**2 / np.multiply.outer(np.multiply.outer(ave_intens, ave_intens), ave_intens)
        return self.g3

    def marginalize_g3(self, num_shots=1000):
        """Form a complex number.

        Keyword arguments:
        real -- the real part (default 0.0)
        imag -- the imaginary part (default 0.0)
        """

        if self.g3 is None:
            self.g3 = self.get_g3(num_shots=num_shots)
        self.g3_4d = np.zeros(4 * (len(self.weights_2d),))

        k1x, k1y, k2x, k2y, k3x, k3y = np.indices(6 * (self.num_pix,))
        # Check the ordering of elements here
        q1x = k1x - k2x
        q1y = k1y - k2y
        q2x = k2x - k3x
        q2y = k2y - k3y
        q1x -= q1x.min()
        q1y -= q1y.min()
        q2x -= q2x.min()
        q2y -= q2y.min()

        np.add.at(self.g3_4d, tuple([q1x,q1y,q2x,q2y]), self.g3)
        if self.weights_4d is None:
            self.init_weights_4d()
        self.g3_4d[self.weights_4d > 0] /= self.weights_4d[self.weights_4d > 0]

        return self.g3_4d


    def closure_from_structure(self, return_phase=False):
        """Form a complex number.

        Keyword arguments:
        real -- the real part (default 0.0)
        imag -- the imaginary part (default 0.0)
        """
        pseudo_coh_ft_double = np.exp(-1j * (self.qr_product_x + self.qr_product_y) ).sum(2)
        coh_12 = np.multiply.outer(pseudo_coh_ft_double, pseudo_coh_ft_double)
        sum_q_x = -np.add.outer(self.q_pix[0,:,:], self.q_pix[0,:,:])
        sum_q_y = -np.add.outer(self.q_pix[1,:,:], self.q_pix[1,:,:])
        sumqr_product_x = np.multiply.outer(sum_q_x, self.coords[0,:])
        sumqr_product_y = np.multiply.outer(sum_q_y, self.coords[1,:])
        coh_1plus2 = np.exp(-1j * ( sumqr_product_x + sumqr_product_y )).sum(4)

        # Output is 4-dimensional matrix
        if return_phase:
            return np.angle(coh_12 * coh_1plus2)
        else:
            if self.weights_4d is None:
                self.init_weights_4d()
            c = 2. * np.real(coh_12 * coh_1plus2)
            c = c / self.num_atoms**3 * (self.weights_4d > 0)
            return c


    def closure_from_data(self, num_shots=1000):
        """Form a complex number.

        Keyword arguments:
        real -- the real part (default 0.0)
        imag -- the imaginary part (default 0.0)
        """
        self.marginalize_g3(num_shots=num_shots)
        if self.g2_2d is None:
            self.marginalize_g2(num_shots=num_shots)

        g1sq = self.g2_2d - 1 + 1./self.num_atoms
        dim = 2*self.num_pix-1
        qx, qy = np.indices(2*(dim,))
        q12x = np.add.outer(qx, qx)
        q12x -= (dim)//2
        q12x[ (q12x < 0) | (q12x >= dim) ] = 0
        q12y = np.add.outer(qy, qy)
        q12y -= (dim)//2
        q12y[ (q12y < 0) | (q12y >= dim) ] = 0
        n = self.num_atoms

        weights = self.weights_4d

        c = ((self.g3_4d - (1 - 3/n + 4/n**2)
             - (1-2/n)*(np.add.outer(g1sq, g1sq)+g1sq[q12x,q12y]))
             * (weights > 0))
        return c

    def phase_from_structure(self):
        """Form a complex number.

        Keyword arguments:
        real -- the real part (default 0.0)
        imag -- the imaginary part (default 0.0)
        """
        return self.closure_from_structure(return_phase=True)


    def phase_from_data(self, num_shots=1000):
        """Form a complex number.

        Keyword arguments:
        real -- the real part (default 0.0)
        imag -- the imaginary part (default 0.0)
        """
        clos = self.closure_from_data(num_shots=num_shots)
        clos = clos / 2

        # Remove magnitude of the g1 product
        g1sq = self.g2_2d - 1 + 1./self.num_atoms
        g1sq[g1sq < 0] = 0.00000000001
        g1 = np.sqrt(g1sq)
        dim = 2*self.num_pix-1
        qx, qy = np.indices(2*(dim,))
        q12x = np.add.outer(qx, qx)
        q12x -= (dim)//2
        q12x[ (q12x < 0) | (q12x >= dim) ] = 0
        q12y = np.add.outer(qy, qy)
        q12y -= (dim)//2
        q12y[ (q12y < 0) | (q12y >= dim) ] = 0

        clos = clos / (np.multiply.outer(g1, g1) * g1[q12x, q12y])
        clos[np.abs(clos) > 1] = np.sign(clos[np.abs(clos) > 1])

        phase = np.arccos(clos)

        return phase


    def cosPhi_from_structure(self):
        """Form a complex number.

        Keyword arguments:
        real -- the real part (default 0.0)
        imag -- the imaginary part (default 0.0)
        """
        real_phase = self.coh_phase_double[self.num_pix-1:, self.num_pix-1:]
        Phi = np.zeros(4*(self.num_pix,))
        for n in range(self.num_pix):
            for m in range(self.num_pix):
                # Phi(n1,n2,m1,m2)
                Phi[n,m,:,:] = np.abs( np.roll(np.roll(real_phase, -n, axis=0), -m, axis=1) - real_phase - real_phase[n,m] )
        #Phi = Phi[:self.num_pix//2+1, :self.num_pix//2+1, :self.num_pix//2+1, :self.num_pix//2+1]
        return np.cos(Phi)


    def cosPhi_fft(self, num_shots=1000):
        """Form a complex number.

        Keyword arguments:
        real -- the real part (default 0.0)
        imag -- the imaginary part (default 0.0)
        """
        g2 = self.marginalize_g2(num_shots=num_shots)
        g1sq = g2 - 1. + 1./self.num_atoms
        g1sq[g1sq < 0] = 0.0000000000001
        g1sq_reduced = g1sq[self.num_pix // 2:3 * self.num_pix // 2, self.num_pix // 2:3 * self.num_pix // 2]
        g1_reduced = np.sqrt(g1sq_reduced)
        if self.g3_4d is None:
            self.g3_4d = self.marginalize_g3(num_shots=num_shots)

        dim = self.num_pix
        qx, qy = np.indices(2 * (dim,))
        q12x = np.add.outer(qx, qx)
        q12x -= (dim) // 2
        q12x[(q12x < 0) | (q12x >= dim)] = 0
        q12y = np.add.outer(qy, qy)
        q12y -= (dim) // 2
        q12y[(q12y < 0) | (q12y >= dim)] = 0
        n = self.num_atoms

        if self.weights_4d is None:
            self.init_weights_4d()
        weights = self.weights_4d[self.num_pix // 2:3 * self.num_pix // 2, self.num_pix // 2:3 * self.num_pix // 2, self.num_pix // 2:3 * self.num_pix // 2, self.num_pix // 2:3 * self.num_pix // 2]
        c = (self.g3_4d - (1 - 3 / n + 4 / n**2) - (1 - 2 / n) * (np.add.outer(g1sq_reduced, g1sq_reduced) + g1sq_reduced[q12x, q12y])) * (weights>0)

        clos = c/2

        # Remove magnitude of the g1 product
        clos = clos / (np.multiply.outer(g1_reduced, g1_reduced) * g1_reduced[q12x, q12y])
        clos[np.abs(clos) > 1] = np.sign(clos[np.abs(clos) > 1])

        # Cut down the relevant area
        clos = clos[self.num_pix // 2:, self.num_pix // 2:, self.num_pix // 2:, self.num_pix // 2:]
        # Apply symmetry restrictions
        clos[0,0,:,:] = 1
        clos[:,:,0,0] = 1
        clos = (clos[:,:,:,:] + np.transpose(clos, axes=[2,3,0,1]))/2

        return clos




    def cosPhi_from_data(self, num_shots=1000):
        """Form a complex number.

        Keyword arguments:
        real -- the real part (default 0.0)
        imag -- the imaginary part (default 0.0)
        """
        idx1 = self.num_pix//2
        idx2 = np.mgrid[0:self.num_pix//2+1, 0:self.num_pix//2+1]
        idx2 = idx2[:,::-1, ::-1]

        # Calculate intensity pattern from g2
        g1sq = self.marginalize_g2(num_shots=num_shots) - 1. + 1. / self.num_atoms
        g1sq[g1sq < 0] = 0.0000000001  # It still comes out negative for some reason occasionally, correcting here for now
        g1sq_reduced = g1sq[self.num_pix // 2:3 * self.num_pix // 2, self.num_pix // 2:3 * self.num_pix // 2]  # Always > 0
        g1_reduced = np.sqrt(g1sq_reduced)
        g1 = np.sqrt(g1sq)

        # Calculate slice of g3
        g3 = np.zeros((self.num_pix // 2 + 1, self.num_pix // 2 + 1, self.num_pix, self.num_pix))
        ave_intens = np.zeros(2*(self.num_pix,))
        for i in range(num_shots):
             incoh = self.get_incoh_intens()
             g3 += np.multiply.outer(incoh[idx2[0,:,:], idx2[1,:,:]], incoh) * incoh[idx1, idx1]
             ave_intens += incoh
        g3 *= num_shots**2 / (ave_intens[idx1, idx1] * np.multiply.outer(ave_intens[idx2[0,:,:], idx2[1,:,:]], ave_intens))
        g3 = g3[:, :, self.num_pix//2:, self.num_pix//2:]
        n = self.num_atoms

        # Calculate cos(Phi)
        exp1 = g1sq_reduced[idx2[0,:,:], idx2[1,:,:]][np.newaxis, np.newaxis, :, :]
        idx_test1 = np.add.outer(np.mgrid[0:self.num_pix//2+1, 0:self.num_pix//2+1][0,:,:], np.mgrid[0:self.num_pix//2+1, 0:self.num_pix//2+1][0,:,:])
        idx_test2 = np.add.outer(np.mgrid[0:self.num_pix//2+1, 0:self.num_pix//2+1][1,:,:], np.mgrid[0:self.num_pix//2+1, 0:self.num_pix//2+1][1,:,:])
        exp2 = g1sq[idx_test1+self.num_pix-1,idx_test2+self.num_pix-1]
        exp3 = g1sq_reduced[idx2[0,:,:], idx2[1,:,:]][:,:, np.newaxis, np.newaxis]
        cosPhi = g3 - (1 - 3 / n + 4 / n**2) - (1 - 2 / n) * (exp1 + exp2 + exp3)

        exp4 = g1_reduced[idx2[0,:,:], idx2[1,:,:]][np.newaxis, np.newaxis, :, :]
        exp5 = g1[idx_test1+self.num_pix-1, idx_test2+self.num_pix-1]
        exp6 = g1_reduced[idx2[0,:,:], idx2[1,:,:]][:, :, np.newaxis, np.newaxis]
        cosPhi /= 2 * exp4 * exp5 * exp6
        # What happens if you just don't divide by the denominator if denominator is zero?

        # Apply symmetry restrictions
        cosPhi = (cosPhi[:, :, :, :] + np.transpose(cosPhi, axes=[2, 3, 0, 1])) / 2

        #RANGE CORRECTION
        cosPhi[cosPhi > 1] = 1
        cosPhi[cosPhi < -1] = -1

        return cosPhi



    def PhiSolver(self, num_shots=1000, error_reject=-10):
        """Form a complex number.

        Keyword arguments:
        real -- the real part (default 0.0)
        imag -- the imaginary part (default 0.0)
        """
        cosPhi_from_dataPhase = np.cos(
            self.phase_from_data(num_shots=num_shots))
        cosPhi_from_dataPhase = (cosPhi_from_dataPhase[self.num_pix - 1:2 * self.num_pix,
                                 self.num_pix - 1:2 * self.num_pix,
                                  self.num_pix - 1:2 * self.num_pix,
                                 self.num_pix - 1:2 * self.num_pix] + cosPhi_from_dataPhase[0:self.num_pix, 0:self.num_pix, 0:self.num_pix, 0:self.num_pix][::-1, ::-1, ::-1, ::-1])/2
        cosPhi = cosPhi_from_dataPhase
        # cosPhi = self.cosPhi_from_structure()
        Phi = np.arccos(cosPhi)
        real_phase = self.coh_phase_double[self.num_pix - 1:3 * self.num_pix // 2, self.num_pix - 1:3 * self.num_pix // 2]

        solved = np.zeros(2*(self.num_pix,))
        solved[0,1] = real_phase[0,1]
        solved[1,0] = real_phase[1,0]

        error = np.zeros_like(solved)

        n = 3
        diagonal_flag = 0
        suspect_num = -1 # Index for list of suspect pixels to be picked as alternates in re-solving
        num_pixels = 1 # To re-solve
        perm_num = -1 # The index in the list of permutations to use for alternates in re-solving
        perm = np.zeros(self.num_pix)
        while n < len(Phi[0,0,0,:])+1:
            # Generate list of points across the diagonal to be solved this round
            prev_solve_1 = np.arange(n-1)
            prev_solve_2 = prev_solve_1[::-1]
            prev_solve = np.asarray([prev_solve_1, prev_solve_2])

            to_solve_1 = np.arange(n)
            to_solve_2 = to_solve_1[::-1]
            to_solve = np.asarray([to_solve_1, to_solve_2])

            for m in range(len(to_solve[0,:])):
                current_pair = to_solve[:,m]
                # Generate matrix of indices which fill the box defined by the origin and our current point
                # Find pairs of vectors which span the box and sum to the current vector
                A = np.indices((current_pair[0]+1,current_pair[1]+1))
                B = np.indices((current_pair[0]+1,current_pair[1]+1))
                B[0,:,:] = current_pair[0] - B[0,:,:]
                B[1,:,:] = current_pair[1] - B[1,:,:]
                # Flatten in to list of pairs and remove trivial (0,0) + (n,m) pairs
                A = A.reshape((2,-1))
                B = B.reshape((2,-1))
                A = A[:,1:-1]
                B = B[:,1:-1]

                plus = np.empty((len(A[0,:])))
                minus = np.empty((len(A[0,:])))
                for i in range(len(A[0,:])):
                    # Find the positive and negative solutions
                    plus[i] = Phi[A[0,i],A[1,i],B[0,i],B[1,i]] + solved[A[0,i],A[1,i]] + solved[B[0,i],B[1,i]]
                    minus[i] = -Phi[A[0,i],A[1,i],B[0,i],B[1,i]] + solved[A[0,i],A[1,i]] + solved[B[0,i],B[1,i]]

                theta1 = np.append(plus,minus)
                theta2 = np.append(minus,plus)

                xdata = np.cos(theta1)
                ydata = np.sin(theta2)

                print(current_pair)
                # If error flag has been triggered for the next diagonal, use the alternate value for trial positions
                #next_phi, error_val = self.find_next_phi(xdata=xdata, ydata=ydata)
                if diagonal_flag == n+1 and perm[m] == 1:
                    next_phi, error_val = self.find_next_phi(xdata=xdata, ydata=ydata, AltReturn = True)
                else:
                    next_phi, error_val = self.find_next_phi(xdata=xdata, ydata=ydata)

                solved[current_pair[0],current_pair[1]] = next_phi
                error[current_pair[0],current_pair[1]] = error_val

            # Loop mechanics
            # Reject any solution with a pixel that has error above error_reject
            if np.any(error[to_solve[0,:], to_solve[1,:]] > error_reject):
            #if (np.any( np.abs(np.subtract.outer(error[to_solve[0,:], to_solve[1,:]], error[prev_solve[0,:], prev_solve[1,:]])) > 15) and n>3):
                print("Prev errors: ", error[prev_solve[0,:], prev_solve[1,:]])
                print("Current errors: ", error[to_solve[0,:], to_solve[1,:]])
                print(np.abs(np.subtract.outer(error[to_solve[0,:], to_solve[1,:]], error[prev_solve[0,:], prev_solve[1,:]])))
                diagonal_flag = n
                print("Unacceptable Error! Re-solving previous diagonal.")
                # First, attempt to change pixels adjacent to pixel in current diagonal with the largest error
                print("Errors: ", error[to_solve[0,:], to_solve[1,:]])
                err_idx = np.argmax(error[to_solve[0,:], to_solve[1,:]])
                suspects = np.zeros((4, diagonal_flag-1)) # The fourth row is just a dummy case, only need 3 permutations for a 1 pixel error
                suspects[0, err_idx] = 1
                suspects[1, err_idx-1] = 1
                suspects[2, err_idx-1:err_idx+1] = 1
                suspect_num += 1
                print("Suspect pixels: ", suspects)
                perm = suspects[suspect_num, :]

                # But if that fails, sort through all possible permutations
                if suspect_num > 2:
                    suspect_num = 2
                    elements = np.zeros(diagonal_flag-1)
                    elements[:num_pixels] = 1
                    perms = np.asarray(list(set(permutations(elements))))
                    perm_num += 1
                    if perm_num >= len(perms[:,0]):
                        print("Adding additional pixel to re-solve.")
                        num_pixels += 1
                        elements[:num_pixels] = 1
                        perms = np.asarray(list(set(permutations(elements))))
                        perm_num = 0
                    # In case we have already been through every possible permutation and still not met the error threshold
                    if num_pixels > len(elements):
                        print("WARNING! WARNING!")
                        print("WARNING! WARNING!")
                        print("WARNING! WARNING!")
                        print("WARNING! WARNING!")
                        print("WARNING! WARNING!")
                        print("Every possible permutation of alternate theta have been tested but the error threshold is still exceed.")
                        print("The error threshold is either too stringent or intervention from the user is needed.")
                        # Then, go back to the default case (no alternates) and proceed anyways.
                        # For now, just exit.
                        exit(1)


                    print(perms)
                    perm = perms[perm_num, :]
                n -= 2 # This is outside the "if suspect_num > 2:" statement
            elif diagonal_flag == n:
                diagonal_flag = 0
                suspect_num = -1
                perm_num = -1
                perm = np.zeros(self.num_pix)
            print("suspect_num", suspect_num)
            print("perm_num", perm_num)
            n += 1

        # Solve out to q_max, at this point error resolving should not be needed
        for n in range(1, len(Phi[0,0,0,:])):
            # Generate list of points across the diagonal to be solved this round
            to_solve_1 = np.arange(len(Phi[0,0,0,:]) - n) + n
            to_solve_2 = to_solve_1[::-1]

            to_solve = np.asarray([to_solve_1, to_solve_2])

            for m in range(len(to_solve[0,:])):
                current_pair = to_solve[:,m]
                # Generate matrix of indices which fill the box defined by the origin and our current point
                # Find pairs of vectors which span the box and sum to the current vector
                #current_pair[np.argmin(current_pair)] += 1
                #current_pair[np.argmax(current_pair)] -=1
                #A = np.indices(current_pair)
                #B = np.indices(current_pair)
                A = np.mgrid[0:current_pair[0]+1, 0:current_pair[1]+1]
                B = np.mgrid[0:current_pair[0]+1, 0:current_pair[1]+1]
                B[0,:,:] = current_pair[0] - B[0,:,:]
                B[1,:,:] = current_pair[1] - B[1,:,:]
                # Flatten in to list of pairs and remove trivial (0,0) + (n,m) pairs
                A = A.reshape((2,-1))
                B = B.reshape((2,-1))
                A = A[:,1:-1]
                B = B[:,1:-1]

                plus = np.empty((len(A[0,:])))
                minus = np.empty((len(A[0,:])))
                for i in range(len(A[0,:])):
                    # Find the positive and negative solutions
                    plus[i] = Phi[A[0,i],A[1,i],B[0,i],B[1,i]] + solved[A[0,i],A[1,i]] + solved[B[0,i],B[1,i]]
                    minus[i] = -Phi[A[0,i],A[1,i],B[0,i],B[1,i]] + solved[A[0,i],A[1,i]] + solved[B[0,i],B[1,i]]

                theta1 = np.append(plus,minus)
                theta2 = np.append(minus,plus)

                xdata = np.cos(theta1)
                ydata = np.sin(theta2)

                print(current_pair)
                next_phi, error_val = self.find_next_phi(xdata=xdata, ydata=ydata)

                solved[current_pair[0],current_pair[1]] = next_phi
                error[current_pair[0], current_pair[1]] = error_val


        return solved, error


    def PhiSolver_manualSelect(self, Phi = None, quadX0 = [0,0], Alt = None):
        """Form a complex number.

        Keyword arguments:
        real -- the real part (default 0.0)
        imag -- the imaginary part (default 0.0)
        """
        #real_phase = self.coh_phase_double[self.num_pix - 1:3 * self.num_pix // 2, self.num_pix - 1:3 * self.num_pix // 2]
        real_phase = self.coh_phase_double[self.num_pix //2:self.num_pix , self.num_pix-1: 3*self.num_pix//2][::-1,:]


        solved = np.zeros(2*(self.num_pix,))
        #solved[0,1] = real_phase[0,1]
        #solved[1,0] = real_phase[1,0]
        solved[0, 1] = quadX0[0]
        solved[1, 0] = quadX0[1]

        error = np.zeros_like(solved)

        n = 3
        while n < len(Phi[0,0,0,:])+1:
            # Generate list of points across the diagonal to be solved this round
            #prev_solve_1 = np.arange(n-1)
            #prev_solve_2 = prev_solve_1[::-1]
            #prev_solve = np.asarray([prev_solve_1, prev_solve_2])

            to_solve_1 = np.arange(n)
            to_solve_2 = to_solve_1[::-1]
            to_solve = np.asarray([to_solve_1, to_solve_2])

            for m in range(len(to_solve[0,:])):
                current_pair = to_solve[:,m]
                # Generate matrix of indices which fill the box defined by the origin and our current point
                # Find pairs of vectors which span the box and sum to the current vector
                A = np.indices((current_pair[0]+1,current_pair[1]+1))
                B = np.indices((current_pair[0]+1,current_pair[1]+1))
                B[0,:,:] = current_pair[0] - B[0,:,:]
                B[1,:,:] = current_pair[1] - B[1,:,:]
                # Flatten in to list of pairs and remove trivial (0,0) + (n,m) pairs
                A = A.reshape((2,-1))
                B = B.reshape((2,-1))
                A = A[:,1:-1]
                B = B[:,1:-1]

                plus = np.empty((len(A[0,:])))
                minus = np.empty((len(A[0,:])))
                for i in range(len(A[0,:])):
                    # Find the positive and negative solutions
                    plus[i] = Phi[A[0,i],A[1,i],B[0,i],B[1,i]] + solved[A[0,i],A[1,i]] + solved[B[0,i],B[1,i]]
                    minus[i] = -Phi[A[0,i],A[1,i],B[0,i],B[1,i]] + solved[A[0,i],A[1,i]] + solved[B[0,i],B[1,i]]

                theta1 = np.append(plus,minus)
                theta2 = np.append(minus,plus)

                xdata = np.cos(theta1)
                ydata = np.sin(theta2)

                print(current_pair)
                # If an alternate has been requested by the user for the pixel, choose the other value
                if Alt[current_pair[0],current_pair[1]] == 1:
                    next_phi, error_val = self.find_next_phi(xdata=xdata, ydata=ydata, AltReturn = True)
                else:
                    next_phi, error_val = self.find_next_phi(xdata=xdata, ydata=ydata)

                solved[current_pair[0],current_pair[1]] = next_phi
                error[current_pair[0],current_pair[1]] = error_val
            n += 1

        # Solve phase out to q_max, at this point error resolving should not be needed
        for n in range(1, len(Phi[0,0,0,:])):
            # Generate list of points across the diagonal to be solved this round
            to_solve_1 = np.arange(len(Phi[0,0,0,:]) - n) + n
            to_solve_2 = to_solve_1[::-1]

            to_solve = np.asarray([to_solve_1, to_solve_2])

            for m in range(len(to_solve[0,:])):
                current_pair = to_solve[:,m]
                # Generate matrix of indices which fill the box defined by the origin and our current point
                # Find pairs of vectors which span the box and sum to the current vector
                #current_pair[np.argmin(current_pair)] += 1
                #current_pair[np.argmax(current_pair)] -=1
                #A = np.indices(current_pair)
                #B = np.indices(current_pair)
                A = np.mgrid[0:current_pair[0]+1, 0:current_pair[1]+1]
                B = np.mgrid[0:current_pair[0]+1, 0:current_pair[1]+1]
                B[0,:,:] = current_pair[0] - B[0,:,:]
                B[1,:,:] = current_pair[1] - B[1,:,:]
                # Flatten in to list of pairs and remove trivial (0,0) + (n,m) pairs
                A = A.reshape((2,-1))
                B = B.reshape((2,-1))
                A = A[:,1:-1]
                B = B[:,1:-1]

                plus = np.empty((len(A[0,:])))
                minus = np.empty((len(A[0,:])))
                for i in range(len(A[0,:])):
                    # Find the positive and negative solutions
                    plus[i] = Phi[A[0,i],A[1,i],B[0,i],B[1,i]] + solved[A[0,i],A[1,i]] + solved[B[0,i],B[1,i]]
                    minus[i] = -Phi[A[0,i],A[1,i],B[0,i],B[1,i]] + solved[A[0,i],A[1,i]] + solved[B[0,i],B[1,i]]

                theta1 = np.append(plus,minus)
                theta2 = np.append(minus,plus)

                xdata = np.cos(theta1)
                ydata = np.sin(theta2)

                print(current_pair)
                next_phi, error_val = self.find_next_phi(xdata=xdata, ydata=ydata)

                solved[current_pair[0],current_pair[1]] = next_phi
                error[current_pair[0], current_pair[1]] = error_val

        return solved, error


    def find_next_phi(self, xdata = None, ydata = None, AltReturn = False):
        """Form a complex number.

        Keyword arguments:
        real -- the real part (default 0.0)
        imag -- the imaginary part (default 0.0)
        """
        # Samples the error function and starts minimization near the minimum

        def thetaError(theta):
            return np.minimum((np.add.outer(xdata, -np.cos(theta)))**2, (np.add.outer(ydata, -np.sin(theta)))**2).sum(0)

        def logThetaError(theta):
            return np.log(np.minimum((np.add.outer(xdata, -np.cos(theta)))**2, (np.add.outer(ydata, -np.sin(theta)))**2).sum(0))

        def ABError(AB):
            return np.log(np.minimum((np.add.outer(xdata, -AB[0,:,:]))**2, (np.add.outer(ydata, -AB[1,:,:]))**2).sum(0))

        def opt_func(theta):
            if np.abs(theta) > np.pi:
                return 1e10
            else:
                return np.log(np.sum(np.minimum((xdata - np.cos(theta))**2, (ydata-np.sin(theta))**2)))

        # This error function has negative poles at the solution
        # Search for points theta that have a very large second derivative to find the poles
        theta = np.linspace(-np.pi,np.pi,50000)
        thetaError = thetaError(theta)
        logThetaError = logThetaError(theta)
        dthetaError= np.gradient(logThetaError,theta)
        ddthetaError = np.gradient(dthetaError,theta)
        num_theta = 2 # Number of candidates to accept. Two is optimal.
        mask = (np.argpartition(ddthetaError,-num_theta)[-num_theta:]) # Indices where second derivative is maximal

        # Why not just brute force calculate the minimum of the error function?
        # Why was calculating the second derivative necessary?
        #mask = (np.argpartition(logThetaError, num_theta)[:num_theta])
        print("Possible Theta = ", theta[mask])
        theta0 = theta[mask]

        # Optimize candidate theta and choose the theta with smallest error
        fCandidate = []
        thetaCandidate = []
        for val in theta0:
            res = optimize.minimize(opt_func, x0=val, method='CG', tol=1e-10, options={'gtol':1e-8, 'maxiter':10000})
            fCandidate.append(res.fun)
            thetaCandidate.append(res.x)
        fCandidate = np.asarray(fCandidate)
        print("Error = ", fCandidate)
        thetaCandidate = np.asarray(thetaCandidate)
        thetaFinal = thetaCandidate[np.argmin(fCandidate)]
        fFinal = np.min(fCandidate)
        print("Final Theta = ", thetaFinal)

        if AltReturn:
            thetaFinal = thetaCandidate[np.argmax(fCandidate)]
            fFinal = np.max(fCandidate)
            print("Alternate Triggered!")
            print("Final Theta = ", thetaFinal)

        # Plot some stuff for troubleshooting
        # AB = np.mgrid[-1:1:1j * 500, -1:1:1j * 500]
        # ABError = ABError(AB)
        #
        # import pylab as P
        # fig = P.figure(figsize=(15,5))
        # ax1 = fig.add_subplot(131)
        # ax1.scatter(xdata, ydata)
        # ax1.axvline(x=np.cos(thetaFinal))
        # ax1.axhline(y=np.sin(thetaFinal))
        # ax1.set_xlabel(r"$\cos(\theta)$")
        # ax1.set_ylabel(r"$\sin(\theta)$")
        #
        # ax2 = fig.add_subplot(132)
        # ax2.plot(theta, thetaError/np.abs(thetaError).max(), label = "Error Function")
        # ax2.plot(theta, dthetaError/np.abs(dthetaError).max(), label = "First Derivative")
        # ax2.plot(theta, ddthetaError/np.abs(ddthetaError).max(), label = "Second Derivative")
        # ax2.set_xlabel(r'$\theta$')
        # ax2.set_ylabel("Error Function")
        # ax2.legend()
        #
        # ax3 = fig.add_subplot(133)
        # im = ax3.imshow(ABError, origin='lower', extent=[-1,1,-1,1])
        # ax3.set_xlabel(r"$\cos(\theta)$")
        # ax3.set_ylabel(r"$\sin(\theta)$")
        # P.colorbar(im, ax=ax3)
        # P.tight_layout()
        # P.show()

        # # Plot some stuff for publication
        # import pylab as P
        # fig = P.figure(figsize=(10, 5))
        # P.rcParams.update({'font.size': 22})
        # ax1 = fig.add_subplot(121)
        # ax1.axvline(x=np.cos(thetaFinal), color='r', zorder=1)
        # ax1.axhline(y=np.sin(thetaFinal), color='r', zorder=2)
        # ax1.scatter(xdata[:len(xdata)//2], ydata[:len(xdata)//2], zorder=3, c = 'green')
        # ax1.scatter(xdata[len(xdata) // 2:], ydata[len(xdata) // 2:], zorder=3, c = 'purple')
        # ax1.set_xlabel(r"$\cos\left(\theta_\pm \right)$")
        # ax1.set_ylabel(r"$\sin \left(\theta_\mp \right)$")
        # ax1.text(0.05, 0.95, 'A', transform=ax1.transAxes,
        #          fontsize=22, fontweight='bold', va='top', c='black')
        #
        # ax2 = fig.add_subplot(122)
        # ax2.plot(theta, thetaError , label=r"$E(\phi)$")
        # ax2.plot(theta, logThetaError , label=r"$\log \left[E(\phi)\right]$")
        # ax2.set_xlabel(r'$\phi$')
        # ax2.set_xticks([-np.pi,0,np.pi])
        # ax2.set_xticklabels([r'$-\pi$', '0', r'$\pi$'])
        # #ax2.set_ylabel("Error")
        # ax2.text(0.05, 0.95, 'B', transform=ax2.transAxes,
        #          fontsize=22, fontweight='bold', va='top', c='black')
        # P.rcParams.update({'font.size': 16})
        # ax2.legend(loc='lower right')
        # P.tight_layout()
        # P.show()

        # Return ideal phi and the value of the error function at that phi
        return np.arctan2(np.sin(thetaFinal), np.cos(thetaFinal)), fFinal