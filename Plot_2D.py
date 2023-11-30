#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as P
import Speckle_2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker
from skimage.restoration import unwrap_phase
from scipy import optimize
from mpl_point_clicker import clicker
import matplotlib.gridspec as gridspec
from matplotlib import colors


class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
             self.format = r'$\mathdefault{%s}$' % self.format

class Plot_2D:
    def __init__(self, num_atoms=3, num_pix=201, kmax=25, useCrystal = False, useDFT = False):
        self.num_atoms = num_atoms
        self.num_pix = num_pix
        self.kmax = kmax
        self.useDFT = useDFT
        self.fluo = Speckle_2D.Fluorescence_2D(kmax=kmax, num_pix=num_pix,
                                               num_atoms=num_atoms)


    def plot_Object(self):
        box_extent = np.max(self.fluo.x_pix[0])
        fig = P.figure(figsize=(15, 10))
        ax1 = fig.add_subplot(231)
        ax1.set_title("Object")
        ax1.imshow(self.fluo.object, extent=[-box_extent,box_extent,-box_extent,box_extent], origin='lower')

        obj_NoPhase = np.fft.fftshift(self.fluo.coh_ft)
        obj_NoPhase = np.abs(obj_NoPhase)
        obj_Phase = np.fft.fftshift(self.fluo.coh_ft)
        phase = np.fft.fftshift(self.fluo.coh_phase)
        obj_Phase = np.abs(obj_Phase) * np.exp(1j * phase)

        obj_NoPhase = np.fft.ifftn(obj_NoPhase)
        obj_Phase = np.fft.ifftn(obj_Phase)

        obj_NoPhase = np.fft.fftshift(obj_NoPhase)
        if not self.fluo.useDFT:
            obj_Phase = np.fft.fftshift(obj_Phase) # When using DFT mode, why does this line need commenting?

        box_extent = np.max(np.fft.fftshift(np.fft.fftfreq(self.fluo.num_pix, d=2 * self.fluo.kmax / self.fluo.num_pix)))
        ax2 = fig.add_subplot(232)
        ax2.set_title("Object from Intensity + Phase")
        # Not sure why this needs to be transposed below, but it does give the correct image
        ax2.imshow(np.abs(obj_Phase), extent=[-box_extent,box_extent,-box_extent,box_extent], origin='lower')
        ax2.set_xlim([-1, 1])
        ax2.set_ylim([-1, 1])
        ax3 = fig.add_subplot(233)
        ax3.set_title("Object from Intensity")
        ax3.imshow(np.abs(obj_NoPhase), extent=[-box_extent,box_extent,-box_extent,box_extent], origin='lower')
        ax3.set_xlim([-1, 1])
        ax3.set_ylim([-1, 1])

        # Double resolution
        box_extent = np.max(self.fluo.x_pix[0])
        ax4 = fig.add_subplot(234)
        ax4.set_title("Object")
        ax4.imshow(self.fluo.object_double, extent=[-box_extent, box_extent, -box_extent, box_extent], origin='lower')

        obj_NoPhase = np.fft.fftshift(self.fluo.coh_ft_double)
        obj_NoPhase = np.abs(obj_NoPhase)
        obj_Phase = np.fft.fftshift(self.fluo.coh_ft_double)
        phase = np.fft.fftshift(self.fluo.coh_phase_double)
        obj_Phase = np.abs(obj_Phase) * np.exp(1j * phase)

        obj_NoPhase = np.fft.ifftn(obj_NoPhase)
        obj_Phase = np.fft.ifftn(obj_Phase)

        obj_NoPhase = np.fft.fftshift(obj_NoPhase)
        if not self.fluo.useDFT:
            obj_Phase = np.fft.fftshift(obj_Phase)  # When using DFT mode, why does this line need commenting?

        box_extent = np.max(np.fft.fftshift(np.fft.fftfreq(self.fluo.num_pix, d=2 * self.fluo.kmax / self.fluo.num_pix)))
        ax5 = fig.add_subplot(235)
        ax5.set_title("Object from Intensity + Phase")
        # Not sure why this needs to be transposed below, but it does give the correct image
        ax5.imshow(np.abs(obj_Phase), extent=[-box_extent, box_extent, -box_extent, box_extent], origin='lower')
        ax5.set_xlim([-1,1])
        ax5.set_ylim([-1,1])
        ax6 = fig.add_subplot(236)
        ax6.set_title("Object from Intensity")
        ax6.imshow(np.abs(obj_NoPhase), extent=[-box_extent, box_extent, -box_extent, box_extent], origin='lower')
        ax6.set_xlim([-1, 1])
        ax6.set_ylim([-1, 1])

        P.tight_layout()
        P.show()


    def plot_Shot(self):
        # PLOT DETECTOR INTENSITIES, SINGLE SHOT
        fig = P.figure(figsize=(10,5))
        s = fig.add_subplot(121)
        s.set_title(("Coherent Intensity"))
        im = s.imshow(np.abs(self.fluo.coh_ft)**2)
        P.colorbar(im, ax=s)

        s = fig.add_subplot(122)
        s.set_title(("Incoherent (Fluorescence) Intensity"))
        im = s.imshow(self.fluo.get_incoh_intens())
        P.colorbar(im, ax=s)

        fig.suptitle("Single Shot Field Intensity at Detector")
        P.tight_layout()
        P.show()


    def plot_Intensities(self, num_shots=10000):
        # PLOT DETECTOR INTENSITIES, g2 MARGINALIZED
        colormap = 'viridis'

        fig = P.figure(figsize=(10, 10))
        s = fig.add_subplot(221)
        s.set_title(("True Intensity"))
        true = np.abs(self.fluo.coh_ft)**2
        im = s.imshow(true, cmap=colormap, origin='lower')
        P.colorbar(im, ax=s)

        s = fig.add_subplot(222)
        s.set_title(("Intensity Measured via g2"))
        measured = self.fluo.marginalize_g2(num_shots=num_shots) - 1 + 1. / self.fluo.num_atoms
        im = s.imshow(measured, cmap=colormap, origin='lower')
        P.colorbar(im, ax=s)

        s = fig.add_subplot(223)
        s.set_title(("Sum of Speckle Intensities"))
        incoh_sum = np.zeros(2*(self.fluo.num_pix,))
        for n in range(num_shots):
            incoh_sum += self.fluo.get_incoh_intens()
        im = s.imshow(incoh_sum/num_shots, cmap=colormap, vmax=1, vmin=0, origin='lower')
        P.colorbar(im, ax=s)

        s = fig.add_subplot(224)
        s.set_title(("Phase Noise (True-Measured)"))
        # You can't compare to coh_ft_double here, the double resolution object is not the same object!
        measured = measured[self.fluo.num_pix//2:3*self.fluo.num_pix//2, self.fluo.num_pix//2:3*self.fluo.num_pix//2]
        im = s.imshow(true - measured, cmap=colormap, origin='lower')#, vmax=1, vmin=0)
        P.colorbar(im, ax=s)

        fig.suptitle("Field Intensity at Detector")
        P.tight_layout()
        P.show()


    def plot_Closure(self, num_shots=10000, saveMem = False):
        # PLOT THE CLOSURES AND THEIR DIFFERENCE
        dim = (2 * self.fluo.num_pix - 1)//2
        cstruct = self.fluo.closure_from_structure()[dim, :, dim, :]
        if saveMem:
            dim = dim//2
        cdata = self.fluo.closure_from_data(num_shots=num_shots)[dim, :, dim, :]

        # Plot
        fig = P.figure(figsize=(15, 5))
        s = fig.add_subplot(131)
        im = s.imshow(cdata)
        s.set_title("Closure from Data")
        P.colorbar(im, ax=s)

        s = fig.add_subplot(132)
        im = s.imshow(cstruct)
        s.set_title("Closure from Structure")
        P.colorbar(im, ax=s)

        s = fig.add_subplot(133)
        if saveMem:
            cstruct = cstruct[self.fluo.num_pix // 2: 3 * self.fluo.num_pix // 2, self.fluo.num_pix // 2: 3 * self.fluo.num_pix // 2]
        im = s.imshow(cdata - cstruct)
        s.set_title("Difference")
        P.colorbar(im, ax=s)
        P.tight_layout()
        P.show()


    def plot_ClosurePhase(self, num_shots = 10000, saveMem = False):
        # Plot a slice of the closure phase
        dim = (2*self.fluo.num_pix-1)//2
        cPhaseStruct = self.fluo.phase_from_structure()[dim,:,dim,:]
        if saveMem:
            dim = dim//2
        cPhaseData = self.fluo.phase_from_data(num_shots=num_shots)[dim, :, dim, :]

        # Plot
        fig = P.figure(figsize=(15, 5))
        s = fig.add_subplot(131)
        im = s.imshow(cPhaseData)
        s.set_title("Phase from Data")
        P.colorbar(im, ax=s)

        s = fig.add_subplot(132)
        im = s.imshow(np.abs(cPhaseStruct))
        s.set_title("Phase from Structure")
        P.colorbar(im, ax=s)

        s = fig.add_subplot(133)
        if saveMem:
            cPhaseStruct = cPhaseStruct[self.fluo.num_pix // 2: 3 * self.fluo.num_pix // 2, self.fluo.num_pix // 2: 3 * self.fluo.num_pix // 2]
        im = s.imshow(np.arccos(np.cos(cPhaseData)-np.cos(cPhaseStruct) ) )
        s.set_title("arccos(cos(Phi_data) - cos(Phi_Struct))")
        P.colorbar(im, ax=s)
        P.tight_layout()
        P.show()



    def plot_cosPhi(self, num_shots=10000):
        # PLOT THE cosPhi MAP FOR ERROR COMPARISON
        cosPhi_from_structure = self.fluo.cosPhi_from_structure()[1,:,1,:]
        cosPhi_from_data = self.fluo.cosPhi_from_data(num_shots=num_shots)[1,:,1,:]
        cosPhi_from_dataPhase = np.cos(
            self.fluo.phase_from_data(num_shots=num_shots))
        cosPhi_from_dataPhase = ( cosPhi_from_dataPhase[self.fluo.num_pix-1:2*self.fluo.num_pix,self.fluo.num_pix-1:2*self.fluo.num_pix,self.fluo.num_pix-1:2*self.fluo.num_pix,self.fluo.num_pix-1:2*self.fluo.num_pix] + cosPhi_from_dataPhase[0:self.fluo.num_pix,0:self.fluo.num_pix,0:self.fluo.num_pix,0:self.fluo.num_pix][::-1,::-1,::-1,::-1] )[1,:,1,:]/2
        cosPhi_from_structurePhase = np.cos(self.fluo.phase_from_structure())[1,:,1,:]

        fig = P.figure(figsize=(20, 5))
        s = fig.add_subplot(151)
        im = s.imshow(cosPhi_from_data, origin="lower")
        s.set_title("cos(Phi) from Data")
        P.colorbar(im, ax=s)

        s = fig.add_subplot(153)
        im = s.imshow(cosPhi_from_structure, origin="lower")
        s.set_title("cos(Phi) from Structure")
        P.colorbar(im, ax=s)

        s = fig.add_subplot(154)
        im = s.imshow(cosPhi_from_dataPhase, origin="lower")
        s.set_title("cos(Phi) from phase_from_data")
        P.colorbar(im, ax=s)

        s = fig.add_subplot(155)
        im = s.imshow(cosPhi_from_structurePhase, origin="lower")
        s.set_title("cos(Phi) from phase_from_structure")
        P.colorbar(im, ax=s)
        P.tight_layout()
        P.show()

    def plot_PhiSolver_manualSelect(self, num_shots = 1000, altLabel=False):
        # Manually select troublesome pixels
        quad1_real_phase = self.fluo.coh_phase_double[self.num_pix - 1: ,
                      self.num_pix - 1:]
        quad2_real_phase = self.fluo.coh_phase_double[:self.num_pix, self.num_pix-1:][::-1,:]

        cosPhi_from_dataPhase = np.cos(
            self.fluo.phase_from_data(num_shots=num_shots))
        quad1_cosPhi_from_dataPhase = (cosPhi_from_dataPhase[self.num_pix - 1:2 * self.num_pix,
                                 self.num_pix - 1:2 * self.num_pix,
                                 self.num_pix - 1:2 * self.num_pix,
                                 self.num_pix - 1:2 * self.num_pix] + cosPhi_from_dataPhase[0:self.num_pix,
                                                                      0:self.num_pix, 0:self.num_pix, 0:self.num_pix][
                                                                      ::-1, ::-1, ::-1, ::-1]) / 2
        quad2_cosPhi_from_dataPhase = (cosPhi_from_dataPhase[
                                 0: self.num_pix,
                                 self.num_pix - 1:2 * self.num_pix, 0:self.num_pix,
                                 self.num_pix - 1:2 * self.num_pix,] + cosPhi_from_dataPhase[ self.num_pix - 1:2 * self.num_pix,
                                0:self.num_pix, self.num_pix - 1:2 * self.num_pix, 0:self.num_pix][::-1, ::-1, ::-1, ::-1]) / 2

        quad1_cosPhi = quad1_cosPhi_from_dataPhase
        quad2_cosPhi = quad2_cosPhi_from_dataPhase[::-1,:,::-1,:]
        # cosPhi = self.fluo.cosPhi_from_structure()
        quad1_Phi = np.arccos(quad1_cosPhi)
        quad2_Phi = np.arccos(quad2_cosPhi)

        quad1_solved = None
        quad2_solved = None
        quad1_error = None
        quad2_error = None

        quad1_alternates = np.zeros(2*(self.num_pix,))
        quad2_alternates = np.zeros(2*(self.num_pix,))

        # Manually solve quadrant 1
        xAlt = 0  # X position of user-labeled alternates
        yAlt = 0  # Y position of user-labeled alternates
        while (xAlt is not None) & (yAlt is not None):
            X0 = [quad1_real_phase[0,1], quad1_real_phase[1,0]]
            quad1_solved, quad1_error = self.fluo.PhiSolver_manualSelect(Phi=quad1_Phi, quadX0=X0, Alt=quad1_alternates)

            fig = P.figure(figsize=(7, 7))
            P.rcParams.update({'font.size': 22})
            # Plot the error
            ax1 = fig.add_subplot(221)
            im = ax1.imshow(quad1_error, cmap='coolwarm', origin="lower")
            P.colorbar(im, ax=ax1)
            klicker = clicker(ax1, ["event"], markers=["x"])
            ax1.set_title("Error "+str(int(np.sum(quad1_error))))
            # Plot the alternates
            ax2 = fig.add_subplot(222)
            im = ax2.imshow(quad1_alternates, cmap='gray', origin="lower")
            P.colorbar(im, ax=ax2)
            ax2.set_title("Alternate")
            # Plot solved phase
            ax3 = fig.add_subplot(223)
            im = ax3.imshow(quad1_solved, cmap='viridis', origin="lower")
            P.colorbar(im, ax=ax3)
            ax3.set_title("Solved")
            # Plot real phase
            ax4 = fig.add_subplot(224)
            im = ax4.imshow(quad1_real_phase, cmap='viridis', origin="lower")
            P.colorbar(im, ax=ax4)
            ax4.set_title("Quad 1")
            P.show()

            if klicker.get_positions()['event'].size != 0:
                xAlt = np.round(klicker.get_positions()['event'][:, 1]).astype(int)
                yAlt = np.round(klicker.get_positions()['event'][:, 0]).astype(int)

                print("xAlt", xAlt, "yAlt", yAlt)
                if (quad1_alternates[xAlt, yAlt] == 1).any():
                    quad1_alternates[xAlt, yAlt] = 0
                else:
                    quad1_alternates[xAlt, yAlt] = 1
            else:
                xAlt = None
                yAlt = None


        # Manually solve quadrant 2
        xAlt = 0  # X position of user-labeled alternates
        yAlt = 0  # Y position of user-labeled alternates
        while (xAlt is not None) & (yAlt is not None):
            X0 = [quad2_real_phase[0, 1], quad2_real_phase[1, 0]]
            quad2_solved, quad2_error = self.fluo.PhiSolver_manualSelect(Phi=quad2_Phi, quadX0=X0, Alt=quad2_alternates)

            fig = P.figure(figsize=(7, 7))
            P.rcParams.update({'font.size': 22})
            # Plot the error
            ax1 = fig.add_subplot(221)
            im = ax1.imshow(quad2_error, cmap='coolwarm', origin="lower")
            P.colorbar(im, ax=ax1)
            klicker = clicker(ax1, ["event"], markers=["x"])
            ax1.set_title("Error " + str(int(np.sum(quad2_error))))
            # Plot the alternates
            ax2 = fig.add_subplot(222)
            im = ax2.imshow(quad2_alternates, cmap='gray', origin="lower")
            P.colorbar(im, ax=ax2)
            ax2.set_title("Alternate")
            # Plot solved phase
            ax3 = fig.add_subplot(223)
            im = ax3.imshow(quad2_solved, cmap='viridis', origin="lower")
            P.colorbar(im, ax=ax3)
            ax3.set_title("Solved")
            # Plot real phase
            ax4 = fig.add_subplot(224)
            im = ax4.imshow(quad2_real_phase, cmap='viridis', origin="lower")
            P.colorbar(im, ax=ax4)
            ax4.set_title("Quad 2")
            P.show()

            if klicker.get_positions()['event'].size != 0:
                xAlt = np.round(klicker.get_positions()['event'][:, 1]).astype(int)
                yAlt = np.round(klicker.get_positions()['event'][:, 0]).astype(int)

                print("xAlt", xAlt, "yAlt", yAlt)
                if (quad2_alternates[xAlt, yAlt] == 1).any():
                    quad2_alternates[xAlt, yAlt] = 0
                else:
                    quad2_alternates[xAlt, yAlt] = 1
            else:
                xAlt = None
                yAlt = None

        # Show final complete result
        error = np.zeros_like(self.fluo.coh_phase_double)
        solved = np.zeros_like(self.fluo.coh_phase_double)
        true = self.fluo.coh_phase_double

        solved[self.num_pix - 1:, self.num_pix - 1:] = quad1_solved
        solved[:self.num_pix , self.num_pix-1:] = quad2_solved[::-1,:]
        solved[:,:self.num_pix] = -solved[:,self.num_pix-1:][::-1,::-1]

        error[self.num_pix - 1:, self.num_pix - 1:] = quad1_error
        error[:self.num_pix , self.num_pix-1:] = quad2_error[::-1,:]
        error[:,:self.num_pix] = error[:,self.num_pix-1:][::-1,::-1]

        box_extent = np.max(np.fft.fftshift(np.fft.fftfreq(2*self.fluo.num_pix, d=2 * self.fluo.kmax / self.fluo.num_pix)))
        measured_amplitude = self.fluo.marginalize_g2(
            num_shots=num_shots) - 1 + 1. / self.fluo.num_atoms
        measured_amplitude[measured_amplitude<0] = 0

        obj_solved = np.abs(np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(measured_amplitude*np.exp(1j*solved)))))
        obj_true = np.abs(np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(measured_amplitude * np.exp(1j * true)))))

        outer = gridspec.GridSpec(1, 2, width_ratios=[3.7, 1], wspace=0.4)
        gs1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], wspace=0.4)
        gs2 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[1])
        fig = P.figure(figsize=(16,4))
        P.rcParams.update({'font.size': 16})
        # Plot full solved
        ax1 = fig.add_subplot(gs1[0]) #141)
        im = ax1.imshow(solved, cmap='twilight_shifted', origin="lower", extent=[-2*self.kmax,2*self.kmax,-2*self.kmax,2*self.kmax], vmin=-np.pi, vmax=np.pi)
        divider = make_axes_locatable(ax1)
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        cbar = P.colorbar(im, cax=cax1)
        ax1.set_aspect('equal')
        ax1.get_xaxis().set_visible(False)
        #ax1.get_yaxis().set_visible(False)
        cbar.set_ticks([-np.pi, 0, np.pi])
        cbar.set_ticklabels([r"$-\pi$", "0", r"$\pi$"])
        if not altLabel:
            ax1.text(0.05, 0.95, 'A', transform=ax1.transAxes,
                     fontsize=22, fontweight='bold', va='top', c='white')
        if altLabel:
            ax1.text(0.05, 0.95, 'E', transform=ax1.transAxes,
                     fontsize=22, fontweight='bold', va='top', c='white')
            ax1.get_xaxis().set_visible(True)
        ax1.set_xlabel(r"$k_x$ $[\mathrm{Length}]^{-1}$")
        ax1.set_ylabel(r"$k_y$ $[\mathrm{Length}]^{-1}$")
        if not altLabel:
            ax1.set_title(r"$\phi_{\mathrm{Solved}}$")
        ax1.set_xticks([-4, 0, 4])
        ax1.set_yticks([-4, 0, 4])
        # Plot full true
        # ax2 = fig.add_subplot(145)
        # im = ax2.imshow(true, cmap='twilight_shifted', origin="lower", extent=[-2*self.kmax,2*self.kmax,-2*self.kmax,2*self.kmax], vmin=-np.pi, vmax=np.pi)
        # divider = make_axes_locatable(ax2)
        # cax2 = divider.append_axes("right", size="5%", pad=0.05)
        # cbar = P.colorbar(im, cax=cax2)
        # ax2.set_aspect('equal')
        # #ax2.get_xaxis().set_visible(False)
        # #ax2.get_yaxis().set_visible(False)
        # cbar.set_ticks([-np.pi, 0, np.pi])
        # cbar.set_ticklabels([r"$-\pi$", "0", r"$\pi$"])
        # ax2.text(0.05, 0.95, 'D', transform=ax2.transAxes,
        #          fontsize=22, fontweight='bold', va='top', c='white')
        # ax2.set_xlabel(r"$k_x$ $[\mathrm{Length}]^{-1}$")
        # ax2.set_ylabel(r"$k_y$ $[\mathrm{Length}]^{-1}$")
        # ax2.set_title(r"$\phi_{\mathrm{True}}$")
        # Plot full error
        ax3 = fig.add_subplot(gs1[1])#142)
        im = ax3.imshow(error, cmap='coolwarm', origin="lower", extent=[-2*self.kmax,2*self.kmax,-2*self.kmax,2*self.kmax])
        divider = make_axes_locatable(ax3)
        cax3 = divider.append_axes("right", size="5%", pad=0.05)
        cbar = P.colorbar(im, cax=cax3)
        ax3.set_aspect('equal')
        ax3.get_xaxis().set_visible(False)
        ax3.get_yaxis().set_visible(False)
        #cbar.set_ticks([-np.pi, 0, np.pi])
        #cbar.set_ticklabels([r"$-\pi$", "0", r"$\pi$"])
        if not altLabel:
            ax3.text(0.05, 0.95, 'B', transform=ax3.transAxes,
                     fontsize=22, fontweight='bold', va='top', c='white')
        if altLabel:
            ax3.text(0.05, 0.95, 'F', transform=ax3.transAxes,
                     fontsize=22, fontweight='bold', va='top', c='white')
            ax3.get_xaxis().set_visible(True)
        ax3.set_xlabel(r"$k_x$ $[\mathrm{Length}]^{-1}$")
        ax3.set_ylabel(r"$k_y$ $[\mathrm{Length}]^{-1}$")
        if not altLabel:
            ax3.set_title(r"$\log \left[E(\phi_{\mathrm{Solved}}) \right]$")
        ax3.set_xticks([-4, 0, 4])
        ax3.set_yticks([-4, 0, 4])
        # Plot difference
        ax4 = fig.add_subplot(gs1[2])#143)
        im = ax4.imshow(true-solved, cmap='PRGn', origin="lower", extent=[-2*self.kmax,2*self.kmax,-2*self.kmax,2*self.kmax], vmin=-2*np.pi, vmax=2*np.pi)
        divider = make_axes_locatable(ax4)
        cax4 = divider.append_axes("right", size="5%", pad=0.05)
        cbar = P.colorbar(im, cax=cax4)
        ax4.set_aspect('equal')
        ax4.get_xaxis().set_visible(False)
        ax4.get_yaxis().set_visible(False)
        cbar.set_ticks([-2*np.pi, 0, 2*np.pi])
        cbar.set_ticklabels([r"$-2\pi$", "0", r"$2\pi$"])
        if not altLabel:
            ax4.text(0.05, 0.95, 'C', transform=ax4.transAxes,
                     fontsize=22, fontweight='bold', va='top', c='black')
        if altLabel:
            ax4.text(0.05, 0.95, 'G', transform=ax4.transAxes,
                     fontsize=22, fontweight='bold', va='top', c='black')
            ax4.get_xaxis().set_visible(True)
        ax4.set_xlabel(r"$k_x$ $[\mathrm{Length}]^{-1}$")
        ax4.set_ylabel(r"$k_y$ $[\mathrm{Length}]^{-1}$")
        if not altLabel:
            ax4.set_title(r"$\phi_{\mathrm{True}} - \phi_{\mathrm{Solved}}$")
        ax4.set_xticks([-4, 0, 4])
        ax4.set_yticks([-4, 0, 4])
        # Plot object from solved
        ax5 = fig.add_subplot(gs2[0])#144)
        # I'm not sure why the axis reversals and transposes are needed here to make the output match the scatter plot
        im = ax5.imshow( (obj_solved/np.max(obj_solved))[:,::-1].T[::-1,:], cmap='hot', origin="lower", extent=[-box_extent,box_extent,-box_extent,box_extent], vmin=0, vmax=1)
        ax5.scatter(self.fluo.coords[0, :], self.fluo.coords[1, :], facecolors='none', edgecolors='cyan')
        divider = make_axes_locatable(ax5)
        cax5 = divider.append_axes("right", size="5%", pad=0.05)
        cbar = P.colorbar(im, cax=cax5)
        ax5.set_aspect('equal')
        ax5.get_xaxis().set_visible(False)
        #ax5.get_yaxis().set_visible(False)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels([r"0", "1"])
        if not altLabel:
            ax5.text(0.05, 0.95, 'D', transform=ax5.transAxes,
                    fontsize=22, fontweight='bold', va='top', c='white')
        if altLabel:
            ax5.text(0.05, 0.95, 'H', transform=ax5.transAxes,
                     fontsize=22, fontweight='bold', va='top', c='white')
            ax5.get_xaxis().set_visible(True)
        ax5.set_xlabel(r"$X$ $[\mathrm{Length}]$")
        ax5.set_ylabel(r"$Y$ $[\mathrm{Length}]$")
        if not altLabel:
            ax5.set_title(r"$ \tilde{\mathcal{F}} \left\{ \left|g^{(1)} \right| e^{i \phi_{\mathrm{Solved}}} \right\}$")
        ax5.set_xticks([-1,0,1])
        ax5.set_yticks([-1, 0, 1])
        # Plot object from true
        # ax6 = fig.add_subplot(246)
        # # I'm not sure why the axis reversals and transposes are needed here to make the output match the scatter plot
        # im = ax6.imshow( (obj_true/np.max(obj_true))[:,::-1].T[::-1,:], cmap='hot', origin="lower", extent=[-box_extent,box_extent,-box_extent,box_extent])
        # ax6.scatter(self.fluo.coords[0, :], self.fluo.coords[1, :])
        # divider = make_axes_locatable(ax6)
        # cax6 = divider.append_axes("right", size="5%", pad=0.05)
        # cbar = P.colorbar(im, cax=cax6)
        # ax6.set_aspect('equal')
        # #ax6.get_xaxis().set_visible(False)
        # #ax6.get_yaxis().set_visible(False)
        # #cbar.set_ticks([-np.pi, 0, np.pi])
        # #cbar.set_ticklabels([r"$-\pi$", "0", r"$\pi$"])
        # ax6.text(0.05, 0.95, 'F', transform=ax6.transAxes,
        #          fontsize=22, fontweight='bold', va='top', c='white')
        # ax6.set_xlabel(r"$X$ $[\mathrm{Length}]$")
        # ax6.set_ylabel(r"$Y$ $[\mathrm{Length}]$")
        # ax6.set_title(r"$ \tilde{\mathcal{F}} \left\{ \left|g^{(1)} \right| e^{i \phi_{\mathrm{True}}} \right\}$")
        # ax6.set_xticks([-1, 0, 1])
        # ax6.set_yticks([-1, 0, 1])
        #P.tight_layout(pad = 0)
        P.show()


    def plot_PhiSolver(self, num_shots = 1000):
        from skimage.restoration import unwrap_phase
        solved, error = self.fluo.PhiSolver(num_shots=num_shots)

        solved = solved
        real_phase = self.fluo.coh_phase_double[self.fluo.num_pix - 1:2 * self.fluo.num_pix -1, self.fluo.num_pix - 1:2 * self.fluo.num_pix -1]
        #real_phase = unwrap_phase(real_phase)

        fig = P.figure(figsize=(10, 10))
        P.rcParams.update({'font.size': 22})
        # Plot the solved phase branch
        ax1 = fig.add_subplot(221)
        im = ax1.imshow(solved, cmap='viridis', origin="lower")
        ax1.set_title(r"$\phi$ Solved")
        divider = make_axes_locatable(ax1)
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        P.colorbar(im, cax=cax1)
        ax1.set_aspect('equal')

        ax2 = fig.add_subplot(222)
        im = ax2.imshow(real_phase, cmap='viridis', origin="lower")
        ax2.set_title(r"$\phi$ Truth")
        divider = make_axes_locatable(ax2)
        cax2 = divider.append_axes("right", size="5%", pad=0.05)
        P.colorbar(im, cax=cax2)
        ax2.set_aspect('equal')

        ax3 = fig.add_subplot(223)
        im = ax3.imshow(error, cmap='coolwarm', origin="lower")
        ax3.set_title(r"$\log \left[E(\phi) \right]$")
        divider = make_axes_locatable(ax3)
        cax3 = divider.append_axes("right", size="5%", pad=0.05)
        P.colorbar(im, cax=cax3)
        ax3.set_aspect('equal')

        ax4 = fig.add_subplot(224)
        im = ax4.imshow(real_phase-solved, cmap='coolwarm', origin="lower")
        ax4.set_title("Truth - Solved")
        divider = make_axes_locatable(ax4)
        cax4 = divider.append_axes("right", size="5%", pad=0.05)
        P.colorbar(im, format=OOMFormatter(-4, mathText=True), cax=cax4)
        ax4.set_aspect('equal')
        P.tight_layout()
        P.subplots_adjust(wspace=0.4, hspace=0.4)
        P.show()


    def learnStructure(self, num_shots = 10000):
        # Initial data to be fitted
        Phi_from_dataPhase = self.fluo.phase_from_data(num_shots=num_shots)
        Phi_from_dataPhase = (Phi_from_dataPhase[self.fluo.num_pix - 1:3 * self.fluo.num_pix // 2,
                                 self.fluo.num_pix - 1:3 * self.fluo.num_pix // 2,
                                 self.fluo.num_pix - 1:3 * self.fluo.num_pix // 2,
                                 self.fluo.num_pix - 1:3 * self.fluo.num_pix // 2]
        + Phi_from_dataPhase[self.fluo.num_pix // 2:self.fluo.num_pix,
                             self.fluo.num_pix // 2:self.fluo.num_pix,
                             self.fluo.num_pix // 2:self.fluo.num_pix,
                             self.fluo.num_pix // 2:self.fluo.num_pix][::-1, ::-1, ::-1, ::-1]) / 2
        g2_from_data = self.fluo.marginalize_g2(num_shots=num_shots)

        # Simple optimization of the error function using differential evolution on compact support
        print("Learning real-space solution...")
        res = optimize.differential_evolution(error_func, bounds=2*self.fluo.num_atoms*[(-1,1),], args=(self.fluo.q_pix, g2_from_data, Phi_from_dataPhase, self.fluo.num_pix), workers=-1)

        print("Solution", res.x.reshape(2,self.fluo.num_atoms))
        print("Actual", self.fluo.coords)

        # Note that we use self.num_atoms here, in case we are working with examples where
        # some parameters of the simulation to be solved are unknown
        self.trial = Speckle_2D.Fluorescence_2D(kmax=self.kmax,
                                                num_pix=self.num_pix,
                                                num_atoms=self.num_atoms,
                                                x=res.x.reshape(2,
                                                                self.num_atoms))

        fig = P.figure(figsize=(14, 7))
        s = fig.add_subplot(251)
        box_extent = np.max(self.fluo.x_pix[0])
        im = s.imshow(self.fluo.object, origin="lower", extent=[-box_extent, box_extent, -box_extent, box_extent])
        s.set_title("Object Truth")
        P.colorbar(im, ax=s)

        # Plotting the object in real space
        s = fig.add_subplot(256)
        box_extent = np.max(self.trial.x_pix[0])
        s.imshow(self.trial.object, origin="lower", extent=[-box_extent, box_extent, -box_extent, box_extent])
        s.set_title("Solution")
        P.colorbar(im, ax=s)

        # Momentum space objects
        s = fig.add_subplot(252)
        box_extent = np.max(self.fluo.q_pix[0])
        s.imshow(np.abs(self.fluo.coh_ft_double), origin="lower", extent=[-box_extent, box_extent, -box_extent, box_extent])
        s.set_title("Object Modulus")
        P.colorbar(im, ax=s)

        s = fig.add_subplot(257)
        box_extent = np.max(self.trial.q_pix[0])
        s.imshow(np.abs(self.trial.coh_ft_double), origin="lower", extent=[-box_extent, box_extent, -box_extent, box_extent])
        s.set_title("Solution Modulus")
        P.colorbar(im, ax=s)

        # Solution Phi
        s = fig.add_subplot(253)
        im = s.imshow(Phi_from_dataPhase[0,:,0,:], origin="lower")
        s.set_title("Phi from True Structure")
        P.colorbar(im, ax=s)

        # Trial Phi
        s = fig.add_subplot(258)
        im = s.imshow(np.arccos(self.trial.cosPhi_from_structure())[:self.num_pix//2+1,:self.num_pix//2+1,:self.num_pix//2+1,:self.num_pix//2+1][0,:,0,:], origin="lower")
        s.set_title("Phi from Trial Structure")
        P.colorbar(im, ax=s)

        # Fourier transforms
        s = fig.add_subplot(254)
        # Object
        obj_NoPhase = np.fft.fftshift(self.fluo.coh_ft)
        obj_NoPhase = np.abs(obj_NoPhase)
        obj_Phase = np.fft.fftshift(self.fluo.coh_ft)
        phase = np.fft.fftshift(self.fluo.coh_phase)
        obj_Phase = np.abs(obj_Phase) * np.exp(1j * phase)
        obj_NoPhase = np.fft.ifftn(obj_NoPhase)
        obj_Phase = np.fft.ifftn(obj_Phase)
        obj_NoPhase = np.fft.fftshift(obj_NoPhase)
        if not self.fluo.useDFT:
            obj_Phase = np.fft.fftshift(obj_Phase)  # When using DFT mode, why does this line need commenting?
        box_extent = np.max(np.fft.fftshift(np.fft.fftfreq(self.fluo.num_pix, d=2 * self.fluo.kmax / self.fluo.num_pix)))
        s.imshow(np.abs(obj_Phase), origin="lower", extent=[-box_extent, box_extent, -box_extent, box_extent])
        s.set_title("True Object from Intensity + Phase")
        P.colorbar(im, ax=s)

        s = fig.add_subplot(259)
        # Trial Solution
        obj_NoPhase = np.fft.fftshift(self.trial.coh_ft)
        obj_NoPhase = np.abs(obj_NoPhase)
        obj_Phase = np.fft.fftshift(self.trial.coh_ft)
        phase = np.fft.fftshift(self.trial.coh_phase)
        obj_Phase = np.abs(obj_Phase) * np.exp(1j * phase)
        obj_NoPhase = np.fft.ifftn(obj_NoPhase)
        obj_Phase = np.fft.ifftn(obj_Phase)
        obj_NoPhase = np.fft.fftshift(obj_NoPhase)
        if not self.trial.useDFT:
            obj_Phase = np.fft.fftshift(obj_Phase)  # When using DFT mode, why does this line need commenting?
        box_extent = np.max(np.fft.fftshift(np.fft.fftfreq(self.trial.num_pix, d=2 * self.trial.kmax / self.trial.num_pix)))
        s.imshow(np.abs(obj_Phase), origin="lower", extent=[-box_extent, box_extent, -box_extent, box_extent])
        s.set_title("Trial Object from Intensity + Phase")
        P.colorbar(im, ax=s)

        s = fig.add_subplot(255)
        box_extent = 2*self.fluo.kmax
        s.imshow(unwrap_phase(self.fluo.coh_phase_double), origin="lower", extent=[-box_extent, box_extent, -box_extent, box_extent])
        s.set_title("True Object Phase")
        P.colorbar(im, ax=s)

        s = fig.add_subplot(2,5,10)
        box_extent = 2 * self.trial.kmax
        s.imshow(unwrap_phase(self.trial.coh_phase_double), origin="lower", extent=[-box_extent, box_extent, -box_extent, box_extent])
        s.set_title("Trial Phase")
        P.colorbar(im, ax=s)

        P.tight_layout()
        P.show()


def error_func(x, q_pix, g2_from_data, Phi_from_dataPhase, num_pix):
    # Error function for fitting structure to the closure phase
    # Update coordinates
    x = x.reshape(2,len(x)//2)
    qr_product_x = np.multiply.outer(q_pix[0, :, :], x[0, :])
    qr_product_y = np.multiply.outer(q_pix[1, :, :], x[1, :])
    coh_ft_double = np.exp(-1j * (qr_product_x + qr_product_y + 0)*2*np.pi).mean(2)
    coh_phase_double = np.angle(coh_ft_double)

    def cosPhi_from_structure():
        real_phase = coh_phase_double[num_pix - 1:, num_pix - 1:]
        Phi = np.zeros(4 * (num_pix,))
        for n in range(num_pix):
            for m in range(num_pix):
                # Phi(n1,n2,m1,m2)
                Phi[n, m, :, :] = np.abs(
                    np.roll(np.roll(real_phase, -n, axis=0), -m, axis=1) - real_phase - real_phase[n, m])
        # Phi = Phi[:self.num_pix//2+1, :self.num_pix//2+1, :self.num_pix//2+1, :self.num_pix//2+1]
        return np.cos(Phi)

    # Update comparison data
    Phi_from_structure = np.arccos(cosPhi_from_structure())[:num_pix//2+1,:num_pix//2+1,:num_pix//2+1,:num_pix//2+1]

    # Number of atoms of the trial solution is a parameter!
    intensity_from_structure = np.abs(coh_ft_double)**2
    intensity_from_data = g2_from_data - 1 + 1 / len(x[0,:])
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
        return np.sum(1000*diffPhiSquare) + np.sum(diffRealSquare) + np.sum(diffIntensSquare)


def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)