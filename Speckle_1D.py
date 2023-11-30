#!/usr/bin/env python

import numpy as np
from numba import jit

class Fluorescence_1D:
    def __init__(self, kmax=10, num_pix=201, num_atoms=4, x=None):
        """Simulate fluorescence speckle from a 1D array of atoms and compute various correlation functions.

        Keyword arguments:
            kmax (float) - maximum coordinate in reciprocal space
            num_pix (int) - number of pixels in reciprocal space, must be an odd number
            num_atoms (int) - number of atoms in the random array, smaller numbers of atoms lead to less HBT phase noise
            x (float) - array of user-supplied coordinates to define custom atom array
        """
        self.kmax = kmax
        self.num_pix = num_pix
        self.num_atoms = num_atoms
        self.x = x # User-supplied coordinates
        self.init_system()

    def init_system(self):
        """Initialize arrays and variables, generate atomic array. Some arrays are not initialized on startup to save resources.
        """
        print("Initializing system...")
        self.k_pix = np.linspace(-self.kmax, self.kmax, self.num_pix)
        self.k_pix_even = np.linspace(-self.kmax, self.kmax, self.num_pix+1)
        self.x_pix = np.linspace(-1, 1, self.num_pix)
        self.x_double_pix = np.linspace(-1,1, 2*self.num_pix-1)
        self.weights = np.correlate(np.ones(self.num_pix), np.ones(self.num_pix), mode = 'full')
        self.q_pix = np.linspace(-2 * self.kmax, 2 * self.kmax, 2 * self.num_pix - 1)
        self.g2 = None
        self.g3 = None
        self.g2_1d = None
        self.g3_2d = None
        self.weights_2d = None

        self.randomize_coords()

    def init_weights_2d(self):
        """Initialize the 2D weights used to marginalize the 3D triple correlation function.
        """
        self.weights_2d = self.compute_weights_2d(num_pix=self.num_pix)

    @staticmethod
    @jit(nopython=True, parallel=False)
    def compute_weights_2d(num_pix=1):
        """Calculate the 2D weights using explicit for-loops.

            Keyword arguments:
                num_pix (int) - the number of pixels, self.num_pix, for the simulation
            """
        weights_2d = np.zeros((2*num_pix-1, 2*num_pix-1))

        for k1 in range(num_pix):
            for k2 in range(num_pix):
                for k3 in range(num_pix):
                    q1 = k1 - k2 + num_pix - 1
                    q2 = k2 - k3 + num_pix - 1
                    weights_2d[q1, q2] += 1

        return weights_2d

    def randomize_coords(self):
        """Form a complex number.

        Keyword arguments:
        real -- the real part (default 0.0)
        imag -- the imaginary part (default 0.0)
        """
        self.coords = np.random.random((self.num_atoms)) * 2 - 1  # Set spatial extent of real space object here

        if self.x is not None:
            self.coords = self.x

        # Define the object for plotting
        # This is a real space object, but we place it in a discrete real space with the same number of bins as k-space for DFT
        self.object = np.zeros_like(self.x_pix)
        self.object[np.digitize(self.coords,self.x_pix)] = 1/self.num_atoms
        self.object_double = np.zeros_like(self.x_double_pix)
        self.object_double[np.digitize(self.coords, self.x_double_pix)] = 1/self.num_atoms
        # object_double is NOT the same object with double sampling, it is slightly different in the binning

        # Define the coherent diffraction
        self.kr_product = np.outer(self.k_pix, self.coords)
        self.kr_product_even = np.outer(self.k_pix_even, self.coords)
        self.qr_product = np.outer(self.q_pix, self.coords)
        self.coh_ft = np.exp(-1j * self.kr_product*2*np.pi).mean(1)
        self.coh_phase = np.angle(self.coh_ft)
        self.coh_ft_double = np.exp(-1j * self.qr_product*2*np.pi).mean(1)
        self.coh_phase_double = np.angle(self.coh_ft_double)


    def get_incoh_intens(self):
        """Form a complex number.

        Keyword arguments:
        real -- the real part (default 0.0)
        imag -- the imaginary part (default 0.0)
        """
        return np.abs(np.exp(-1j * (self.kr_product + np.random.random(self.num_atoms))*2*np.pi).mean(1))**2

    def get_g2(self, num_shots=1000):
        """Form a complex number.

        Keyword arguments:
        real -- the real part (default 0.0)
        imag -- the imaginary part (default 0.0)
        """
        print("Performing second-order intensity correlation using outer product...")
        ave_intens = np.zeros(self.num_pix)
        self.g2 = np.zeros(2 * (self.num_pix,))
        for i in range(num_shots):
            #print("Correlating frame ", i)
            incoh = self.get_incoh_intens()
            self.g2 += np.outer(incoh, incoh)
            ave_intens += incoh
        self.g2 *= (num_shots) / np.outer(ave_intens, ave_intens)
        print("Finished correlation...")
        return self.g2

    def marginalize_g2(self, num_shots=1000):
        """Form a complex number.

        Keyword arguments:
        real -- the real part (default 0.0)
        imag -- the imaginary part (default 0.0)
        """
        if self.g2_1d is None:
            if self.g2 is None:
                self.g2 = self.get_g2(num_shots)
            q_2d = np.subtract.outer(np.arange(self.num_pix), np.arange(self.num_pix))
            q_2d -= q_2d.min()
            self.g2_1d = np.zeros_like(self.weights)
            np.add.at(self.g2_1d, q_2d, self.g2)
            self.g2_1d = self.g2_1d / self.weights
            return self.g2_1d
        else:
            return self.g2_1d


    def get_g3(self, num_shots=1000):
        """Form a complex number.

        Keyword arguments:
        real -- the real part (default 0.0)
        imag -- the imaginary part (default 0.0)
        """
        print("Performing third-order correlation using outer product...")
        ave_intens = np.zeros(self.num_pix)
        self.g3 = np.zeros(3 * (self.num_pix,))
        for i in range(num_shots):
            #print("Correlating frame ", i)
            incoh = self.get_incoh_intens()
            self.g3 += np.multiply.outer(np.outer(incoh, incoh), incoh)
            ave_intens += incoh
        self.g3 *= num_shots**2 / np.multiply.outer(np.outer(ave_intens, ave_intens), ave_intens)
        print("Finished correlation...")
        return self.g3

    def marginalize_g3(self, num_shots=1000):
        """Form a complex number.

        Keyword arguments:
        real -- the real part (default 0.0)
        imag -- the imaginary part (default 0.0)
        """

        if self.g3 is None:
            self.g3 = self.get_g3(num_shots=num_shots)
        self.g3_2d = np.zeros(2 * (len(self.weights),))

        x, y, z = np.indices(3 * (self.num_pix,))
        q1 = x - y
        q1 -= q1.min()
        q2 = y - z
        q2 -= q2.min()

        np.add.at(self.g3_2d, tuple([q1, q2]), self.g3)
        if self.weights_2d is None:
            self.init_weights_2d()
        self.g3_2d[self.weights_2d > 0] /= self.weights_2d[self.weights_2d > 0]

        return self.g3_2d

    def closure_from_structure(self, return_phase=False):
        """Form a complex number.

        Keyword arguments:
        real -- the real part (default 0.0)
        imag -- the imaginary part (default 0.0)
        """
        pseudo_coh_ft_double = np.exp(-1j * self.qr_product).sum(1)
        coh_12 = np.multiply.outer(pseudo_coh_ft_double, pseudo_coh_ft_double)
        sum_q = -np.add.outer(self.q_pix, self.q_pix)
        sumqr_product = np.multiply.outer(sum_q, self.coords)
        coh_1plus2 = np.exp(-1j * sumqr_product).sum(2)

        if return_phase:
            return np.angle(coh_12 * coh_1plus2)
        else:
            c = 2. * np.real(coh_12 * coh_1plus2)
            c = c / self.num_atoms**3 * (self.weights_2d > 0)
            return c


    def closure_from_data(self, num_shots=1000):
        """Form a complex number.

        Keyword arguments:
        real -- the real part (default 0.0)
        imag -- the imaginary part (default 0.0)
        """
        if self.g3_2d is None:
            self.marginalize_g3(num_shots=num_shots)
        if self.g2_1d is None:
            self.marginalize_g2(num_shots=num_shots)

        g1sq = self.g2_1d - 1 + 1. / self.num_atoms
        dim = 2*self.num_pix - 1
        q12 = np.add.outer(np.arange(dim), np.arange(dim))
        q12 -= (dim)//2
        q12[(q12 < 0) | (q12 >= dim)] = 0
        n = self.num_atoms

        weights = self.weights_2d

        c = (self.g3_2d - (1 - 3 / n + 4 / n**2) - (1 - 2 / n) * (
                np.add.outer(g1sq, g1sq) + g1sq[q12])) * (weights > 0)
        return c


    def cosPhi_from_structure(self):
        """Form a complex number.

        Keyword arguments:
        real -- the real part (default 0.0)
        imag -- the imaginary part (default 0.0)
        """
        return np.cos(self.closure_from_structure(return_phase=True))


    def cosPhi_from_data(self, num_shots=1000):
        """Form a complex number.

        Keyword arguments:
        real -- the real part (default 0.0)
        imag -- the imaginary part (default 0.0)
        """
        clos = self.closure_from_data(num_shots=num_shots)
        clos = clos / 2

        # Remove magnitude of the g1 product
        g1sq = self.g2_1d - 1 + 1. / self.num_atoms
        g1sq[g1sq < 0] = 0.00000000001
        g1 = np.sqrt(g1sq)
        dim = 2 * self.num_pix - 1
        q12 = np.add.outer(np.arange(dim), np.arange(dim))
        q12 = q12 - len(g1) // 2
        q12[(q12 < 0) | (q12 >= len(g1))] = 0

        clos = clos / (np.multiply.outer(g1, g1) * g1[q12])
        clos[np.abs(clos) > 1] = np.sign(clos[np.abs(clos) > 1])

        cosPhi = clos

        return cosPhi