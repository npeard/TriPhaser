#!/usr/bin/env python

import numpy as np
import scipy.signal
from scipy import optimize



class Fluorescence_1D:
    def __init__(self, kmax=10, num_pix=201, num_atoms=4, useDFT = False, x = None):
        """Form a complex number.

        Keyword arguments:
        real -- the real part (default 0.0)
        imag -- the imaginary part (default 0.0)
        """
        self.kmax = kmax
        self.num_pix = num_pix
        self.num_atoms = num_atoms
        self.useDFT = useDFT
        self.x = x
        self.init_system()

    def init_system(self):
        """Form a complex number.

        Keyword arguments:
        real -- the real part (default 0.0)
        imag -- the imaginary part (default 0.0)
        """
        self.k_pix = np.linspace(-self.kmax, self.kmax, self.num_pix)
        self.k_pix_even = np.linspace(-self.kmax, self.kmax, self.num_pix+1)
        self.x_pix = np.linspace(-1, 1, self.num_pix)
        self.x_double_pix = np.linspace(-1,1, 2*self.num_pix-1)
        self.weights = np.correlate(np.ones(self.num_pix), np.ones(self.num_pix), mode = 'full')
        self.q_pix = np.linspace(-2 * self.kmax, 2 * self.kmax, 2 * self.num_pix - 1)
        if self.x is None:
            self.init_weights_2d()
        self.randomize_coords()

    def init_weights_2d(self):
        """Form a complex number.

        Keyword arguments:
        real -- the real part (default 0.0)
        imag -- the imaginary part (default 0.0)
        """
        x, y, z = np.indices(3 * (self.num_pix,))
        q1 = x - y
        q1 -= q1.min()
        q2 = y - z
        q2 -= q2.min()
        self.weights_2d = np.zeros(2 * (len(self.weights),))
        np.add.at(self.weights_2d, tuple([q1, q2]), 1)

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
        if self.useDFT:
            self.coh_ft = np.fft.fftshift(np.fft.fft(self.object))
            self.coh_phase = np.angle(self.coh_ft)
            self.coh_ft_double = np.fft.fftshift(np.fft.fft(self.object_double))
            self.coh_phase_double = np.angle(self.coh_ft_double)
        else:
            # This uses the analytical model
            self.coh_ft = np.exp(-1j * self.kr_product*2*np.pi).mean(1)
            self.coh_phase = np.angle(self.coh_ft)
            self.coh_ft_double = np.exp(-1j * self.qr_product*2*np.pi).mean(1)
            self.coh_phase_double = np.angle(self.coh_ft_double)
        self.g2 = None
        self.g3 = None
        self.g2_1d = None
        self.g3_2d = None


    def get_incoh_intens(self):
        """Form a complex number.

        Keyword arguments:
        real -- the real part (default 0.0)
        imag -- the imaginary part (default 0.0)
        """
        if self.useDFT:
            Phases = np.random.random(len(self.object)) * 2 * np.pi
            T = self.object * np.exp(1j * Phases)
            I = np.abs(np.fft.fft(T))**2
            # No fftshift?
            return I
        else:
            return np.abs(np.exp(-1j * (self.kr_product + np.random.random(self.num_atoms))*2*np.pi).mean(1))**2
            # Trapezoidal pixel integration
            # even = np.abs(np.exp(-1j * (self.kr_product_even + np.random.random(self.num_atoms))*2*np.pi).mean(1))**2
            # integrated = ( even[1:] + even[:-1] )/2
            # interval = self.k_pix_even[1:] - self.k_pix_even[:-1]
            # return integrated*interval


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


    def g2_fft(self, num_shots=1000):
        """Form a complex number.

        Keyword arguments:
        real -- the real part (default 0.0)
        imag -- the imaginary part (default 0.0)
        """
        # Uses FFT convolution to obtain the marginalized g2. See cross-correlation theorem.
        print("Performing second-order correlation using FFT...")

        self.g2_1d = np.zeros_like(self.q_pix)

        for i in range(num_shots):
            incoh = self.get_incoh_intens()
            add = scipy.signal.fftconvolve(incoh, incoh[::-1])
            self.g2_1d += add

        self.g2_1d /= self.weights * num_shots / self.num_atoms**2
        print("Finished correlation...")
        return self.g2_1d


    def marginalize_g2(self, num_shots=1000, saveMem=False):
        """Form a complex number.

        Keyword arguments:
        real -- the real part (default 0.0)
        imag -- the imaginary part (default 0.0)
        """
        if saveMem:
            return self.g2_fft(num_shots=num_shots)
        elif self.g2_1d is None:
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


    def get_g3_fft(self, num_shots=1000):
        """Form a complex number.

        Keyword arguments:
        real -- the real part (default 0.0)
        imag -- the imaginary part (default 0.0)
        """
        print("Performing third-order correlation using FFT...")
        self.g3_2d = np.zeros(2*(len(self.k_pix),))
        sum = np.zeros_like(self.g3_2d, dtype=complex)

        q1, q2 = np.indices(2 * (self.num_pix,))
        q3 = (-q1 - q2) % (self.num_pix)
        for i in range(num_shots):
            incoh = self.get_incoh_intens()
            fft_incoh = np.fft.fft(incoh)
            BiSpec = fft_incoh[np.newaxis,:] * fft_incoh[:,np.newaxis] * fft_incoh[q3]
            add = np.fft.ifft2(BiSpec)
            sum += add

        sum = np.fft.fftshift(sum)
        self.g3_2d = np.real(sum[::-1, :])
        # Normalize to match outer product method
        self.g3_2d /= num_shots/self.num_atoms**3
        self.g3_2d /= self.num_pix
        print("Finished correlation...")
        return self.g3_2d


    def marginalize_g3(self, num_shots=1000, saveMem=True):
        """Form a complex number.

        Keyword arguments:
        real -- the real part (default 0.0)
        imag -- the imaginary part (default 0.0)
        """
        if saveMem:
            return self.get_g3_fft(num_shots=num_shots)
        else:
            if self.g3 is None:
                self.g3 = self.get_g3(num_shots=num_shots)
            self.g3_2d = np.zeros(2 * (len(self.weights),))

            x, y, z = np.indices(3 * (self.num_pix,))
            q1 = x - y
            q1 -= q1.min()
            q2 = y - z
            q2 -= q2.min()

            np.add.at(self.g3_2d, tuple([q1, q2]), self.g3)
            self.g3_2d[self.weights_2d > 0] /= self.weights_2d[self.weights_2d > 0]

            return self.g3_2d


    def g1sq_from_g3(self, num_shots=1000):
        """Form a complex number.

        Keyword arguments:
        real -- the real part (default 0.0)
        imag -- the imaginary part (default 0.0)
        """
        if self.g3_2d is None:
            self.g3_2d = self.marginalize_g3(num_shots=num_shots, saveMem=False)

        g3_1d = self.g3_2d[self.num_pix-1,:] # You can also use the other three axes here - they should be equivalent by symmetry
        coeff = [2, 3-6/self.num_atoms, 0, 1-3/self.num_atoms+4/self.num_atoms**2-g3_1d[self.num_pix-1]]
        if np.imag(np.roots(coeff)[2]) == 0:
            g10 = np.real(np.roots(coeff)[2])
        else:
            print("Error: Root of expression should be real but is not.")

        a = 2*g10 + (2-4/self.num_atoms)
        c = (1-2/self.num_atoms)*g10 + (1-3/self.num_atoms+4/self.num_atoms**2) - g3_1d
        g1sq = -c/a

        return g1sq


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


    def closure_from_data(self, num_shots=1000, saveMem = False):
        """Form a complex number.

        Keyword arguments:
        real -- the real part (default 0.0)
        imag -- the imaginary part (default 0.0)
        """
        #if self.g3_2d is None:
        self.marginalize_g3(num_shots=num_shots, saveMem=saveMem)
        if self.g2_1d is None:
            self.marginalize_g2(num_shots=num_shots, saveMem=saveMem)

        g1sq = self.g2_1d - 1 + 1. / self.num_atoms
        dim = 2*self.num_pix - 1
        if saveMem:
            g1sq = g1sq[self.num_pix//2:3*self.num_pix//2]
            dim = self.num_pix
        q12 = np.add.outer(np.arange(dim), np.arange(dim))
        q12 -= (dim)//2
        q12[(q12 < 0) | (q12 >= dim)] = 0
        n = self.num_atoms

        weights = self.weights_2d
        if saveMem:
            weights = self.weights_2d[self.num_pix // 2:3 * self.num_pix // 2, self.num_pix // 2:3 * self.num_pix // 2]

        c = (self.g3_2d - (1 - 3 / n + 4 / n**2) - (1 - 2 / n) * (np.add.outer(g1sq, g1sq) + g1sq[q12])) * (weights > 0)
        return c


    def phase_from_structure(self):
        """Form a complex number.

        Keyword arguments:
        real -- the real part (default 0.0)
        imag -- the imaginary part (default 0.0)
        """
        return self.closure_from_structure(return_phase=True)


    def phase_from_data(self, num_shots=1000, saveMem = False):
        """Form a complex number.

        Keyword arguments:
        real -- the real part (default 0.0)
        imag -- the imaginary part (default 0.0)
        """
        clos = self.closure_from_data(num_shots=num_shots, saveMem=saveMem)
        clos = clos / 2

        # Remove magnitude of the g1 product
        g1sq = self.g2_1d - 1 + 1. / self.num_atoms
        g1sq[g1sq < 0] = 0.00000000001
        g1 = np.sqrt(g1sq)
        dim = 2 * self.num_pix - 1
        if saveMem:
            g1 = g1[self.num_pix // 2:3 * self.num_pix // 2]
            dim = self.num_pix
        q12 = np.add.outer(np.arange(dim), np.arange(dim))
        q12 = q12 - len(g1) // 2
        q12[(q12 < 0) | (q12 >= len(g1))] = 0

        clos = clos / (np.multiply.outer(g1, g1) * g1[q12])
        clos[np.abs(clos) > 1] = np.sign(clos[np.abs(clos) > 1])

        phase = np.arccos(clos)

        return phase



    def cosPhi_from_structure(self):
        """Form a complex number.

        Keyword arguments:
        real -- the real part (default 0.0)
        imag -- the imaginary part (default 0.0)
        """
        real_phase = self.coh_phase_double[self.num_pix - 1:]
        Phi = np.zeros((self.num_pix, self.num_pix))
        for n in range(self.num_pix):
            Phi[n, :] = (np.abs(np.roll(real_phase, -n) - real_phase - real_phase[n]))
        Phi = Phi[:self.num_pix // 2 + 1, :self.num_pix // 2 + 1]

        return np.cos(Phi)



    def cosPhi_fft(self, num_shots=1000):
        """Form a complex number.

        Keyword arguments:
        real -- the real part (default 0.0)
        imag -- the imaginary part (default 0.0)
        """
        g2 = self.marginalize_g2(num_shots=num_shots, saveMem=True)
        g1sq = g2 - 1. + 1. / self.num_atoms
        g1sq[g1sq < 0] = 0.0000000001  # It still comes out negative for some reason occasionally, correcting here for now
        g1sq_reduced = g1sq[self.num_pix // 2:3 * self.num_pix // 2]  # Always > 0
        g1_reduced = np.sqrt(g1sq_reduced)
        if self.g3_2d is None:
            self.g3_2d = self.marginalize_g3(num_shots=num_shots, saveMem=True)
        #g3_2d = self.marginalize_g3(num_shots=num_shots, saveMem=False)[self.num_pix//2:3*self.num_pix//2,self.num_pix//2:3*self.num_pix//2]

        q12 = np.add.outer(np.arange(len(g1sq_reduced)), np.arange(len(g1sq_reduced)))
        q12 = q12 - len(g1sq_reduced) // 2
        q12[(q12 < 0) | (q12 >= len(g1sq_reduced))] = 0
        n = self.num_atoms

        c = (self.g3_2d - (1 - 3 / n + 4 / n**2) - (1 - 2 / n) * (np.add.outer(g1sq_reduced, g1sq_reduced) + g1sq_reduced[q12])) * (
                self.weights_2d[self.num_pix // 2:3 * self.num_pix // 2, self.num_pix // 2:3 * self.num_pix // 2] > 0)

        clos = c / 2

        # Remove magnitude of the g1 product
        clos = clos / (np.multiply.outer(g1_reduced, g1_reduced) * g1_reduced[q12])
        clos[np.abs(clos) > 1] = np.sign(clos[np.abs(clos) > 1])

        # Cut down to the relevant area
        clos = clos[self.num_pix // 2:, self.num_pix // 2:]
        # Apply symmetry restrictions
        clos[0,:] = 1
        clos[:,0] = 1
        clos = (clos+clos.T)/2

        return clos



    def cosPhi_from_data(self, num_shots=1000):
        """Form a complex number.

        Keyword arguments:
        real -- the real part (default 0.0)
        imag -- the imaginary part (default 0.0)
        """
        idx1 = self.num_pix // 2
        idx2 = np.arange(self.num_pix//2+1)[::-1]

        # Calculate intensity pattern from g2
        g1sq = self.marginalize_g2(num_shots=num_shots, saveMem=True) - 1. + 1. / self.num_atoms
        g1sq[g1sq < 0] = 0.0000000001  # It still comes out negative for some reason occasionally, correcting here for now
        g1sq_reduced = g1sq[self.num_pix // 2:3 * self.num_pix // 2]  # Always > 0
        g1_reduced = np.sqrt(g1sq_reduced)
        g1 = np.sqrt(g1sq)

        # Calculate slice of g3
        g3 = np.zeros((self.num_pix//2+1,self.num_pix))
        ave_intens = np.zeros(self.num_pix)
        for i in range(num_shots):
            incoh = self.get_incoh_intens()
            g3 += np.multiply.outer(incoh[idx2], incoh) * incoh[idx1]
            ave_intens += incoh
        g3 *= num_shots**2 / (ave_intens[idx1] * np.multiply.outer(ave_intens[idx2], ave_intens))
        g3 = g3[:,self.num_pix//2:]
        n = self.num_atoms
        #return g3

        # Calculate cos(Phi)
        #ones = np.ones(self.num_pix//2+1)
        #exp1 = np.multiply.outer(ones, g1sq_reduced[idx2])
        exp1 = g1sq_reduced[idx2][:, np.newaxis]
        idx_test = np.add.outer(np.arange(self.num_pix//2+1), np.arange(self.num_pix//2+1))
        exp2 = g1sq[idx_test+self.num_pix-1]
        #exp3 = np.multiply.outer(g1sq_reduced[idx2],ones)
        exp3 = g1sq_reduced[idx2][np.newaxis, :]
        cosPhi = g3 - (1 - 3 / n + 4 / n**2) - (1 - 2 / n) * (exp1 + exp2 + exp3)

        #exp4 = np.multiply.outer(ones, g1_reduced[idx2])
        exp4 = g1_reduced[idx2][:, np.newaxis]
        exp5 = g1[idx_test+self.num_pix-1]
        #exp6 = np.multiply.outer(g1_reduced[idx2],ones)
        exp6 = g1_reduced[idx2][np.newaxis, :]
        cosPhi /= 2 * exp4 * exp5 * exp6

        # ENFORCE SYMMETRY?
        # Is it better to enforce symmetry or extend PhiSolver algorithm to sample the entire Phi map and then average the calculated phases?
        cosPhi = (cosPhi + cosPhi.T)/2

        # RANGE CORRECTION
        cosPhi[cosPhi > 1] = 1
        cosPhi[cosPhi < -1] = -1

        return cosPhi


    def simple_PhiSolver(self, num_shots=1000):
        """Form a complex number.

        Keyword arguments:
        real -- the real part (default 0.0)
        imag -- the imaginary part (default 0.0)
        """
        # Taking into account only "singles and doubles"
        cosPhi = self.cosPhi_from_structure()
        Phi = np.arccos(cosPhi)
        from skimage.restoration import unwrap_phase
        real_phase = unwrap_phase(self.coh_phase_double[self.num_pix - 1:3 * self.num_pix // 2])

        solved = np.zeros(self.num_pix//2)
        solved[1] = real_phase[1]
        print(Phi[:,2])
        for p in range(2,self.num_pix//2):
            sum = 0
            for k in range(1,p):
                sum += Phi[k,1]
            solved[p] = sum + p*solved[1]

        return solved



    def PhiSolver(self, num_shots = 10000):
        """Form a complex number.

        Keyword arguments:
        real -- the real part (default 0.0)
        imag -- the imaginary part (default 0.0)
        """
        cosPhi_from_dataPhase = np.cos(self.phase_from_data(num_shots=num_shots))
        cosPhi_from_dataPhase = (cosPhi_from_dataPhase[self.num_pix - 1:3 * self.num_pix // 2, self.num_pix - 1:3 * self.num_pix // 2] + cosPhi_from_dataPhase[self.num_pix // 2:self.num_pix, self.num_pix // 2:self.num_pix][::-1, ::-1]) / 2
        cosPhi = cosPhi_from_dataPhase
        #cosPhi = self.cosPhi_from_structure()
        Phi = np.arccos(cosPhi)

        # PHI SOLVER ALGORITHM
        # Initial conditions
        solved = np.zeros(self.num_pix)
        error = np.zeros_like(solved)
        real_phase = self.coh_phase_double[self.num_pix - 1:3*self.num_pix//2]
        solved[1] = real_phase[1]

        # Find phi out to KMAX
        error_threshold = 10
        n = 1
        useAlt = False
        while n < self.num_pix//2:
            print("Pixel", n)
            branches = np.zeros((int((n + 3) / 2) - 1, 2))
            for m in range(1, int((n + 3) / 2), 1):
                plus = Phi[n - m + 1, m] + solved[n - m + 1] + solved[m]
                minus = -Phi[n - m + 1, m] + solved[n - m + 1] + solved[m]
                branches[m - 1, 0] = plus
                branches[m - 1, 1] = minus

            theta1 = np.append(branches[:, 0], branches[:, 1])
            theta2 = np.append(branches[:, 1], branches[:, 0])
            xdata = np.cos(theta1)
            ydata = np.sin(theta2)

            next_phi, error_val = self.find_next_phi(xdata = xdata, ydata=ydata, AltReturn = useAlt)
            solved[n+1] = next_phi
            error[n+1] = error_val

            if error[n+1] - error[n] > error_threshold:
                n -= 1
                useAlt = True
            else:
                useAlt = False
                n += 1


        # Find phi out to QMAX
        error_threshold = 10
        n = 0
        #for n in range(0, self.num_pix//2, 1):
        print("QMAX LOOP")
        while n < self.num_pix//2:
            print("Pixel", n+self.num_pix//2)
            branches = np.zeros((int((self.num_pix//2 - n + 3) / 2)-1, 2))
            for m in range(1, int((self.num_pix//2 - n + 3) / 2), 1):
                plus = Phi[self.num_pix//2 - m + 1, m+n] + solved[self.num_pix//2 - m + 1] + solved[m+n]
                minus = -Phi[self.num_pix//2 - m + 1, m+n] + solved[self.num_pix//2 - m + 1] + solved[m+n]
                branches[m - 1, 0] = plus
                branches[m - 1, 1] = minus

            theta1 = np.append(branches[:, 0], branches[:, 1])
            theta2 = np.append(branches[:, 1], branches[:, 0])
            xdata = np.cos(theta1)
            ydata = np.sin(theta2)

            next_phi, error_val = self.find_next_phi(xdata=xdata, ydata=ydata, AltReturn=useAlt)
            solved[n + self.num_pix//2 + 1] = next_phi
            error[n + self.num_pix//2 + 1] = error_val

            if error[n + self.num_pix//2 + 1] - error[n + self.num_pix//2] > error_threshold:
                n -= 1
                useAlt = True
            else:
                useAlt = False
                n += 1

        # Return solved branches
        return solved, error



    def find_next_phi(self, xdata=None, ydata=None, AltReturn = False):
        """Form a complex number.

        Keyword arguments:
        real -- the real part (default 0.0)
        imag -- the imaginary part (default 0.0)
        """
        # Samples the error function and starts minimization near the minimum

        def thetaError(theta):
            return np.minimum((np.add.outer(xdata, -np.cos(theta)))**2, (np.add.outer(ydata, -np.sin(theta)))**2).sum(0)

        def logThetaError(theta):
            return np.log(
                np.minimum((np.add.outer(xdata, -np.cos(theta)))**2, (np.add.outer(ydata, -np.sin(theta)))**2).sum(0))

        def ABError(AB):
            return np.log(
                np.minimum((np.add.outer(xdata, -AB[0, :, :]))**2, (np.add.outer(ydata, -AB[1, :, :]))**2).sum(0))

        def opt_func(theta):
            if np.abs(theta) > np.pi:
                return 1e10
            else:
                return np.log(np.sum(np.minimum((xdata - np.cos(theta))**2, (ydata - np.sin(theta))**2)))

        # This error function has negative poles at the solution
        # Search for points theta that have a very large second derivative to find the poles
        theta = np.linspace(-np.pi, np.pi, 50000)
        thetaError = thetaError(theta)
        logThetaError = logThetaError(theta)
        # dthetaError= np.gradient(thetaError,theta)
        # ddthetaError = np.gradient(dthetaError,theta)
        num_theta = 2  # Number of candidates to accept. Two is optimal.
        # mask = (np.argpartition(ddthetaError,-num_theta)[-num_theta:]) # Indices where second derivative is maximal

        # Why not just brute force calculate the minimum of the error function?
        # Why was calculating the second derivative necessary?
        mask = (np.argpartition(logThetaError, num_theta)[:num_theta])
        print("Possible Theta = ", theta[mask])
        theta0 = theta[mask]

        # Optimize candidate theta and choose the theta with smallest error
        fCandidate = []
        thetaCandidate = []
        for val in theta0:
            res = optimize.minimize(opt_func, x0=val, method='CG', tol=1e-10, options={'gtol': 1e-8, 'maxiter': 10000})
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
        # ax2.plot(theta, logThetaError, label = "Log Error Function")
        # #ax2.plot(theta, dthetaError/np.abs(dthetaError).max(), label = "First Derivative")
        # #ax2.plot(theta, ddthetaError/np.abs(ddthetaError).max(), label = "Second Derivative")
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