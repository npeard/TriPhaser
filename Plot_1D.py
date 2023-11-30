#!/usr/bin/env python

import numpy as np
import pylab as P
import Speckle_1D
from scipy import optimize
import TriPhase_1D

class Plot_1D:
    def __init__(self, num_atoms=5, num_pix=201, kmax=10):
        self.num_atoms = num_atoms
        self.num_pix = num_pix
        self.kmax = kmax
        self.fluo = Speckle_1D.Fluorescence_1D(kmax=self.kmax,
                                               num_pix=self.num_pix,
                                               num_atoms=self.num_atoms)


    def plot_Object(self):
        P.plot(self.fluo.x_pix, self.fluo.object, label="Object")
        print("Coordinates", self.fluo.coords)
        obj_NoPhase = np.fft.fftshift(self.fluo.coh_ft)
        obj_NoPhase = np.abs(obj_NoPhase)
        obj_Phase = np.fft.fftshift(self.fluo.coh_ft)
        phase = np.fft.fftshift(self.fluo.coh_phase)
        obj_Phase = np.abs(obj_Phase) * np.exp(1j * phase)

        obj_NoPhase = np.fft.ifft(obj_NoPhase)
        obj_Phase = np.fft.ifft(obj_Phase)

        obj_NoPhase = np.fft.fftshift(obj_NoPhase)
        if not self.fluo.useDFT:
            obj_Phase = np.fft.fftshift(obj_Phase) # When using DFT mode, why does this line need commenting?

        scaled_x = np.fft.fftshift(np.fft.fftfreq(self.fluo.num_pix,d=2*self.fluo.kmax/self.fluo.num_pix))
        P.plot(scaled_x, np.abs(obj_NoPhase), label="Object from Intensity")
        P.plot(scaled_x, np.abs(obj_Phase), '--', label="Object from Intensity + Phase")
        P.xlim([-1,1])
        P.legend()
        P.tight_layout()
        P.show()


    def plot_Shot(self):
        # PLOT DETECTOR INTENSITIES, SINGLE SHOT
        P.plot(self.fluo.k_pix, np.abs(self.fluo.coh_ft)**2)
        P.plot(self.fluo.k_pix, self.fluo.get_incoh_intens())
        P.title("Single Shot Intensity at Detector")
        P.legend(("Coherent Intensity", "Incoherent (Fluorescence) Intensity"))
        P.tight_layout()
        P.show()


    def plot_Intensities(self, num_shots=10000):
        # PLOT DETECTOR INTENSITIES, g2 MARGINALIZED
        #P.plot(self.fluo.q_pix, np.abs(self.fluo.coh_ft_double)**2, label='True Intensity')
        P.plot(self.fluo.k_pix, np.abs(self.fluo.coh_ft)**2, label='True Intensity')
        g2 = self.fluo.marginalize_g2(num_shots=num_shots)
        measured = (g2 - 1 + 1. / self.fluo.num_atoms)
        P.plot(self.fluo.q_pix, measured, 'o--',
               label=r'Intensity Computed via $g^2$')

        incoh_sum = np.zeros_like(self.fluo.k_pix)
        for n in range(num_shots):
            incoh_sum += self.fluo.get_incoh_intens()
        P.plot(self.fluo.k_pix, incoh_sum / num_shots * self.fluo.num_atoms, label='Summed Speckle Pattern')

        #g1sq_g3 = self.fluo.g1sq_from_g3(num_shots=num_shots)
        #P.plot(self.fluo.q_pix, g1sq_g3, label = 'Intensity Measured via g3')

        P.title("Field Intensity at Detector")
        P.legend()
        P.tight_layout()
        P.show()


    def plot_g2(self, num_shots=10000):
        g2 = self.fluo.get_g2(num_shots=num_shots)
        P.imshow(g2, cmap='gray', origin='lower')
        P.title("g2")
        P.tight_layout()
        P.show()


    def plot_g3(self, num_shots=10000):
        # Compare g3 from outer product and FFT convolution
        fig = P.figure(figsize=(5, 5))
        ax1 = fig.add_subplot(111)
        ax1.set_title(("Outer Product g3"))
        g3 = self.fluo.marginalize_g3(num_shots=num_shots)
        im = ax1.imshow(g3)
        P.colorbar(im, ax=ax1)

        P.tight_layout()
        P.show()


    def plot_Closure(self, num_shots=10000):
        # PLOT THE CLOSURES AND THEIR DIFFERENCE
        fig = P.figure(figsize=(15, 5))
        cdata = self.fluo.closure_from_data(num_shots=num_shots)
        s = fig.add_subplot(131)
        im = s.imshow(cdata)
        s.set_title("Closure from Data")
        P.colorbar(im, ax=s)

        s = fig.add_subplot(132)
        cstruct = self.fluo.closure_from_structure()
        im = s.imshow(self.fluo.closure_from_structure())
        s.set_title("Closure from Structure")
        P.colorbar(im, ax=s)

        s = fig.add_subplot(133)
        im = s.imshow(cdata - cstruct)
        s.set_title("Difference")
        P.colorbar(im, ax=s)
        P.tight_layout()
        P.show()


    def plot_ClosurePhase(self, num_shots=10000):
        # PLOT THE PHASE MAP IN K-SPACE
        fig = P.figure(figsize=(10, 5))
        s = fig.add_subplot(121)
        im = s.imshow(self.fluo.phase_from_data(num_shots=num_shots))
        s.set_title("Phase from Data")
        P.colorbar(im, ax=s)

        s = fig.add_subplot(122)
        im = s.imshow(np.arccos(self.fluo.cosPhi_from_structure()))
        im = s.imshow(np.abs(self.fluo.phase_from_structure()))
        s.set_title("Phase from Structure")
        P.colorbar(im, ax=s)
        P.tight_layout()
        P.show()


    def plot_cosPhi(self, num_shots=10000):
        # PLOT THE cosPhi MAP FOR ERROR COMPARISON
        cosPhi_from_dataPhase = self.fluo.cosPhi_from_data(num_shots=num_shots)
        cosPhi_from_dataPhase = (cosPhi_from_dataPhase[self.fluo.num_pix-1:3*self.fluo.num_pix//2,self.fluo.num_pix-1:3*self.fluo.num_pix//2] + cosPhi_from_dataPhase[self.fluo.num_pix//2:self.fluo.num_pix, self.fluo.num_pix//2:self.fluo.num_pix][::-1,::-1])/2  # Averaging data from both sides of the central axis
        cosPhi_from_structurePhase = self.fluo.cosPhi_from_structure()[
        self.fluo.num_pix-1:,self.fluo.num_pix-1:]

        fig = P.figure(figsize=(8, 4))
        s = fig.add_subplot(121)
        im = s.imshow(cosPhi_from_dataPhase, origin="lower")
        s.set_title("cosPhi from phase_from_data")
        P.colorbar(im, ax=s)

        s = fig.add_subplot(122)
        im = s.imshow(cosPhi_from_structurePhase, origin="lower")
        s.set_title("cosPhi from phase_from_structure")
        P.colorbar(im, ax=s)

        P.tight_layout()
        P.show()

    def plot_simple_PhiSolve(self, num_shots = 1000):
        cosPhi = self.fluo.cosPhi_from_data(num_shots=num_shots)
        initial_phase = self.fluo.coh_phase_double[self.num_pix - 1:3 *
                                                               self.num_pix
                                                               // 2][1]
        solved = TriPhase_1D.simple_PhiSolver(cosPhi,
                                              initial_phase=initial_phase)
        from skimage.restoration import unwrap_phase
        solved = unwrap_phase(solved)
        real_phase = self.fluo.coh_phase[self.fluo.num_pix//2:]
        real_phase = unwrap_phase(real_phase)

        fig = P.figure(figsize=(7, 7))
        # Plot the solved phase branch
        s = fig.add_subplot(111)
        P.plot(np.linspace(0, len(real_phase), len(real_phase)),
               real_phase, label='Exact')
        P.plot(np.linspace(0, len(solved), len(solved)), solved,
               'o--',label=r'$\Phi = |\Phi|$')
        s.set_ylabel(r'$\phi$')
        s.set_xlabel("Pixel Index")
        P.legend()
        P.tight_layout()
        P.show()


    def plot_PhiSolver(self, num_shots=10000):
        cosPhi = self.fluo.cosPhi_from_data(num_shots=num_shots)
        initial_phase = self.fluo.coh_phase_double[self.num_pix - 1:3 *
                                                                    self.num_pix
                                                                    // 2][1]
        solved, error = TriPhase_1D.PhiSolver(cosPhi,
                                              initial_phase=initial_phase)
        real_phase = self.fluo.coh_phase_double[self.fluo.num_pix - 1:]

        # Unwrap the phase
        from skimage.restoration import unwrap_phase
        plot_solved = unwrap_phase(solved[:])
        plot_real_phase = unwrap_phase(real_phase)

        fig = P.figure(figsize=(5,5))
        # Plot the solved phase branch
        s = fig.add_subplot(111)
        P.plot(np.linspace(0, len(real_phase), len(real_phase)),
               plot_real_phase, 'o--', label='Exact')
        P.plot(np.linspace(0, len(real_phase), len(real_phase)),
               plot_solved, label='Solved')
        P.plot(np.linspace(0, len(real_phase), len(real_phase)),
               np.cos(plot_real_phase-plot_solved), label = 'cos(Diff)')
        P.legend()
        P.tight_layout()
        P.show()


    def learnStructure(self, num_shots = 10000):
        # Initial data to be fitted
        Phi_from_dataPhase = self.fluo.phase_from_data(num_shots=num_shots)
        Phi_from_dataPhase = (Phi_from_dataPhase[self.fluo.num_pix - 1:3 * self.fluo.num_pix // 2, self.fluo.num_pix - 1:3 * self.fluo.num_pix // 2] + Phi_from_dataPhase[self.fluo.num_pix // 2:self.fluo.num_pix,self.fluo.num_pix // 2:self.fluo.num_pix][::-1,::-1]) / 2  # Averaging data from both sides of the central axis
        g2_from_data = self.fluo.marginalize_g2(num_shots=num_shots)

        # Simple optimization of the error function using differential evolution on compact support
        print("Learning real-space solution...")
        res = optimize.differential_evolution(error_func, bounds=self.fluo.num_atoms*[(-1,1),], args=(self.fluo.q_pix, g2_from_data, Phi_from_dataPhase, self.fluo.num_pix), workers=-1)

        # # Remove linear ramp
        # c = np.linspace(-0.136,0.6,5000)
        # test = np.zeros_like(c)
        # for shift in range(5000):
        #     test[shift] = error_func(res.x+c[shift], self.fluo.q_pix, g2_from_data, Phi_from_dataPhase, self.fluo.num_pix)
        # fig = P.figure(figsize=(7, 7))
        # s = fig.add_subplot(111)
        # s.plot(c, test)
        # print(c[np.argmin(test)])
        # P.tight_layout()
        # P.show()

        print("Solution", res.x)
        print("Actual", self.fluo.coords)

        self.trial = Speckle_1D.Fluorescence_1D(kmax=self.kmax,
                                                num_pix=self.num_pix,
                                                num_atoms=self.num_atoms,
                                                x=res.x)

        fig = P.figure(figsize=(14, 7))
        s = fig.add_subplot(231)
        im = s.imshow(Phi_from_dataPhase, origin="lower")
        s.set_title("Phi from Data")
        P.colorbar(im, ax=s)

        # Plotting the object in real space
        s = fig.add_subplot(232)
        s.plot(self.fluo.x_pix, self.fluo.object, label="Object")
        s.plot(self.trial.x_pix, self.trial.object, '--', label="Solution")
        s.set_xlim([-1, 1])
        s.legend()

        # Momentum space objects
        s = fig.add_subplot(233)
        s.plot(self.fluo.q_pix, np.abs(self.fluo.coh_ft_double), label="Truth Modulus")
        s.plot(self.trial.q_pix, np.abs(self.trial.coh_ft_double), '--', label="Solution Modulus")
        s.legend()

        # Trial cosPhi
        s = fig.add_subplot(234)
        im = s.imshow(np.arccos(self.trial.cosPhi_from_structure()), origin="lower")
        s.set_title("Phi from Trial Structure")
        P.colorbar(im, ax=s)

        # Fourier transforms
        s = fig.add_subplot(235)
        # Object
        obj_NoPhase = np.fft.fftshift(self.fluo.coh_ft)
        obj_NoPhase = np.abs(obj_NoPhase)
        obj_Phase = np.fft.fftshift(self.fluo.coh_ft)
        phase = np.fft.fftshift(self.fluo.coh_phase)
        obj_Phase = np.abs(obj_Phase) * np.exp(1j * phase)
        obj_NoPhase = np.fft.ifft(obj_NoPhase)
        obj_Phase = np.fft.ifft(obj_Phase)
        obj_NoPhase = np.fft.fftshift(obj_NoPhase)
        if not self.fluo.useDFT:
            obj_Phase = np.fft.fftshift(obj_Phase)  # When using DFT mode, why does this line need commenting?
        scaled_x = np.fft.fftshift(np.fft.fftfreq(self.fluo.num_pix, d=2 * self.fluo.kmax / self.fluo.num_pix))
        #s.plot(scaled_x, np.abs(obj_NoPhase), label="Object from Intensity")
        s.plot(scaled_x, np.abs(obj_Phase), label="Object from Intensity + Phase")

        # Trial Solution
        obj_NoPhase = np.fft.fftshift(self.trial.coh_ft)
        obj_NoPhase = np.abs(obj_NoPhase)
        obj_Phase = np.fft.fftshift(self.trial.coh_ft)
        phase = np.fft.fftshift(self.trial.coh_phase)
        obj_Phase = np.abs(obj_Phase) * np.exp(1j * phase)
        obj_NoPhase = np.fft.ifft(obj_NoPhase)
        obj_Phase = np.fft.ifft(obj_Phase)
        obj_NoPhase = np.fft.fftshift(obj_NoPhase)
        if not self.trial.useDFT:
            obj_Phase = np.fft.fftshift(obj_Phase)  # When using DFT mode, why does this line need commenting?
        scaled_x = np.fft.fftshift(np.fft.fftfreq(self.trial.num_pix, d=2 * self.trial.kmax / self.trial.num_pix))
        #s.plot(scaled_x, np.abs(obj_NoPhase), label="Object from Intensity")
        s.plot(scaled_x, np.abs(obj_Phase), '--', label="Trial Object from Intensity + Phase")
        s.set_xlim([-1, 1])
        s.legend()

        s = fig.add_subplot(236)
        s.plot(self.fluo.q_pix, unwrap_phase(self.fluo.coh_phase_double), label = "Truth Phase")
        s.plot(self.trial.q_pix, unwrap_phase(self.trial.coh_phase_double), '--', label = "Solution Phase")
        s.legend()

        P.tight_layout()
        P.show()


def error_func(x, q_pix, g2_from_data, Phi_from_dataPhase, num_pix):
    # Error function for fitting structure to the closure phase
    # Update coordinates
    qr_product = np.outer(q_pix, x)
    coh_ft_double = np.exp(-1j * qr_product * 2 * np.pi).mean(1)
    coh_phase_double = np.angle(coh_ft_double)

    def cosPhi_from_structure():
        real_phase = coh_phase_double[num_pix - 1:]
        Phi = np.zeros((num_pix, num_pix))
        for n in range(num_pix):
            Phi[n, :] = (np.abs(np.roll(real_phase, -n) - real_phase - real_phase[n]))
        Phi = Phi[:num_pix // 2 + 1, :num_pix // 2 + 1]

        return np.cos(Phi)

    # Update comparison data
    Phi_from_structure = np.arccos(cosPhi_from_structure())
    # Number of atoms of the trial solution is a parameter!
    intensity_from_structure = np.abs(coh_ft_double)**2
    intensity_from_data = g2_from_data - 1 + 1 / len(x)
    intensity_from_data[intensity_from_data<0] = 0

    diffPhi = Phi_from_structure - Phi_from_dataPhase
    diffPhiSquare = diffPhi**2

    diffIntens = intensity_from_structure - intensity_from_data
    diffIntensSquare = diffIntens**2

    real_from_structure = np.fft.fftshift(np.sqrt(intensity_from_structure))
    real_from_structure = np.abs(real_from_structure)
    real_from_structure = np.fft.ifft(real_from_structure)
    real_from_structure = np.fft.fftshift(real_from_structure)

    real_from_data = np.fft.fftshift(np.sqrt(intensity_from_data))
    real_from_data = np.abs(real_from_data)
    real_from_data = np.fft.ifft(real_from_data)
    real_from_data = np.fft.fftshift(real_from_data)

    # Compare real space structures without phase information
    diffReal = np.abs(real_from_structure) - np.abs(real_from_data)
    diffRealSquare = diffReal**2

    if np.any(np.abs(x) > 1):
        return 1e10
    else:
        return np.sum(100*diffPhiSquare) + np.sum(diffRealSquare) + np.sum(diffIntensSquare)


def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)