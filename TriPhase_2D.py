#!/usr/bin/env python

import numpy as np
from scipy import optimize
from itertools import permutations
import h5py
import Speckle_2D

def PhiSolver(self, num_shots=1000, error_reject=-10):
	"""Form a complex number.

	Keyword arguments:
	real -- the real part (default 0.0)
	imag -- the imaginary part (default 0.0)
	"""
	cosPhi_from_dataPhase = np.cos(
		self.phase_from_data(num_shots=num_shots))
	cosPhi_from_dataPhase = (cosPhi_from_dataPhase[
							 self.num_pix - 1:2 * self.num_pix,
							 self.num_pix - 1:2 * self.num_pix,
							 self.num_pix - 1:2 * self.num_pix,
							 self.num_pix - 1:2 * self.num_pix] + cosPhi_from_dataPhase[
																  0:self.num_pix,
																  0:self.num_pix,
																  0:self.num_pix,
																  0:self.num_pix][
																  ::-1, ::-1,
																  ::-1,
																  ::-1]) / 2
	cosPhi = cosPhi_from_dataPhase
	# cosPhi = self.cosPhi_from_structure()
	Phi = np.arccos(cosPhi)
	real_phase = self.coh_phase_double[self.num_pix - 1:3 * self.num_pix // 2,
				 self.num_pix - 1:3 * self.num_pix // 2]

	solved = np.zeros(2 * (self.num_pix,))
	solved[0, 1] = real_phase[0, 1]
	solved[1, 0] = real_phase[1, 0]

	error = np.zeros_like(solved)

	n = 3
	diagonal_flag = 0
	suspect_num = -1  # Index for list of suspect pixels to be picked as alternates in re-solving
	num_pixels = 1  # To re-solve
	perm_num = -1  # The index in the list of permutations to use for alternates in re-solving
	perm = np.zeros(self.num_pix)
	while n < len(Phi[0, 0, 0, :]) + 1:
		# Generate list of points across the diagonal to be solved this round
		prev_solve_1 = np.arange(n - 1)
		prev_solve_2 = prev_solve_1[::-1]
		prev_solve = np.asarray([prev_solve_1, prev_solve_2])

		to_solve_1 = np.arange(n)
		to_solve_2 = to_solve_1[::-1]
		to_solve = np.asarray([to_solve_1, to_solve_2])

		for m in range(len(to_solve[0, :])):
			current_pair = to_solve[:, m]
			# Generate matrix of indices which fill the box defined by the origin and our current point
			# Find pairs of vectors which span the box and sum to the current vector
			A = np.indices((current_pair[0] + 1, current_pair[1] + 1))
			B = np.indices((current_pair[0] + 1, current_pair[1] + 1))
			B[0, :, :] = current_pair[0] - B[0, :, :]
			B[1, :, :] = current_pair[1] - B[1, :, :]
			# Flatten in to list of pairs and remove trivial (0,0) + (n,m) pairs
			A = A.reshape((2, -1))
			B = B.reshape((2, -1))
			A = A[:, 1:-1]
			B = B[:, 1:-1]

			plus = np.empty((len(A[0, :])))
			minus = np.empty((len(A[0, :])))
			for i in range(len(A[0, :])):
				# Find the positive and negative solutions
				plus[i] = Phi[A[0, i], A[1, i], B[0, i], B[1, i]] + solved[
					A[0, i], A[1, i]] + solved[B[0, i], B[1, i]]
				minus[i] = -Phi[A[0, i], A[1, i], B[0, i], B[1, i]] + solved[
					A[0, i], A[1, i]] + solved[B[0, i], B[1, i]]

			theta1 = np.append(plus, minus)
			theta2 = np.append(minus, plus)

			xdata = np.cos(theta1)
			ydata = np.sin(theta2)

			print(current_pair)
			# If error flag has been triggered for the next diagonal, use the alternate value for trial positions
			# next_phi, error_val = self.find_next_phi(xdata=xdata, ydata=ydata)
			if diagonal_flag == n + 1 and perm[m] == 1:
				next_phi, error_val = self.find_next_phi(xdata=xdata,
														 ydata=ydata,
														 AltReturn=True)
			else:
				next_phi, error_val = self.find_next_phi(xdata=xdata,
														 ydata=ydata)

			solved[current_pair[0], current_pair[1]] = next_phi
			error[current_pair[0], current_pair[1]] = error_val

		# Loop mechanics
		# Reject any solution with a pixel that has error above error_reject
		if np.any(error[to_solve[0, :], to_solve[1, :]] > error_reject):
			# if (np.any( np.abs(np.subtract.outer(error[to_solve[0,:], to_solve[1,:]], error[prev_solve[0,:], prev_solve[1,:]])) > 15) and n>3):
			print("Prev errors: ", error[prev_solve[0, :], prev_solve[1, :]])
			print("Current errors: ", error[to_solve[0, :], to_solve[1, :]])
			print(np.abs(
				np.subtract.outer(error[to_solve[0, :], to_solve[1, :]],
								  error[prev_solve[0, :], prev_solve[1, :]])))
			diagonal_flag = n
			print("Unacceptable Error! Re-solving previous diagonal.")
			# First, attempt to change pixels adjacent to pixel in current diagonal with the largest error
			print("Errors: ", error[to_solve[0, :], to_solve[1, :]])
			err_idx = np.argmax(error[to_solve[0, :], to_solve[1, :]])
			suspects = np.zeros((4,
								 diagonal_flag - 1))  # The fourth row is just a dummy case, only need 3 permutations for a 1 pixel error
			suspects[0, err_idx] = 1
			suspects[1, err_idx - 1] = 1
			suspects[2, err_idx - 1:err_idx + 1] = 1
			suspect_num += 1
			print("Suspect pixels: ", suspects)
			perm = suspects[suspect_num, :]

			# But if that fails, sort through all possible permutations
			if suspect_num > 2:
				suspect_num = 2
				elements = np.zeros(diagonal_flag - 1)
				elements[:num_pixels] = 1
				perms = np.asarray(list(set(permutations(elements))))
				perm_num += 1
				if perm_num >= len(perms[:, 0]):
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
					print(
						"Every possible permutation of alternate theta have been tested but the error threshold is still exceed.")
					print(
						"The error threshold is either too stringent or intervention from the user is needed.")
					# Then, go back to the default case (no alternates) and proceed anyways.
					# For now, just exit.
					exit(1)

				print(perms)
				perm = perms[perm_num, :]
			n -= 2  # This is outside the "if suspect_num > 2:" statement
		elif diagonal_flag == n:
			diagonal_flag = 0
			suspect_num = -1
			perm_num = -1
			perm = np.zeros(self.num_pix)
		print("suspect_num", suspect_num)
		print("perm_num", perm_num)
		n += 1

	# Solve out to q_max, at this point error resolving should not be needed
	for n in range(1, len(Phi[0, 0, 0, :])):
		# Generate list of points across the diagonal to be solved this round
		to_solve_1 = np.arange(len(Phi[0, 0, 0, :]) - n) + n
		to_solve_2 = to_solve_1[::-1]

		to_solve = np.asarray([to_solve_1, to_solve_2])

		for m in range(len(to_solve[0, :])):
			current_pair = to_solve[:, m]
			# Generate matrix of indices which fill the box defined by the origin and our current point
			# Find pairs of vectors which span the box and sum to the current vector
			# current_pair[np.argmin(current_pair)] += 1
			# current_pair[np.argmax(current_pair)] -=1
			# A = np.indices(current_pair)
			# B = np.indices(current_pair)
			A = np.mgrid[0:current_pair[0] + 1, 0:current_pair[1] + 1]
			B = np.mgrid[0:current_pair[0] + 1, 0:current_pair[1] + 1]
			B[0, :, :] = current_pair[0] - B[0, :, :]
			B[1, :, :] = current_pair[1] - B[1, :, :]
			# Flatten in to list of pairs and remove trivial (0,0) + (n,m) pairs
			A = A.reshape((2, -1))
			B = B.reshape((2, -1))
			A = A[:, 1:-1]
			B = B[:, 1:-1]

			plus = np.empty((len(A[0, :])))
			minus = np.empty((len(A[0, :])))
			for i in range(len(A[0, :])):
				# Find the positive and negative solutions
				plus[i] = Phi[A[0, i], A[1, i], B[0, i], B[1, i]] + solved[
					A[0, i], A[1, i]] + solved[B[0, i], B[1, i]]
				minus[i] = -Phi[A[0, i], A[1, i], B[0, i], B[1, i]] + solved[
					A[0, i], A[1, i]] + solved[B[0, i], B[1, i]]

			theta1 = np.append(plus, minus)
			theta2 = np.append(minus, plus)

			xdata = np.cos(theta1)
			ydata = np.sin(theta2)

			print(current_pair)
			next_phi, error_val = self.find_next_phi(xdata=xdata, ydata=ydata)

			solved[current_pair[0], current_pair[1]] = next_phi
			error[current_pair[0], current_pair[1]] = error_val

	return solved, error


def PhiSolver_manualSelect(self, Phi=None, quadX0=[0, 0], Alt=None):
	"""Form a complex number.

	Keyword arguments:
	real -- the real part (default 0.0)
	imag -- the imaginary part (default 0.0)
	"""
	# real_phase = self.coh_phase_double[self.num_pix - 1:3 * self.num_pix // 2, self.num_pix - 1:3 * self.num_pix // 2]
	real_phase = self.coh_phase_double[self.num_pix // 2:self.num_pix,
				 self.num_pix - 1: 3 * self.num_pix // 2][::-1, :]

	solved = np.zeros(2 * (self.num_pix,))
	# solved[0,1] = real_phase[0,1]
	# solved[1,0] = real_phase[1,0]
	solved[0, 1] = quadX0[0]
	solved[1, 0] = quadX0[1]

	error = np.zeros_like(solved)

	n = 3
	while n < len(Phi[0, 0, 0, :]) + 1:
		# Generate list of points across the diagonal to be solved this round
		# prev_solve_1 = np.arange(n-1)
		# prev_solve_2 = prev_solve_1[::-1]
		# prev_solve = np.asarray([prev_solve_1, prev_solve_2])

		to_solve_1 = np.arange(n)
		to_solve_2 = to_solve_1[::-1]
		to_solve = np.asarray([to_solve_1, to_solve_2])

		for m in range(len(to_solve[0, :])):
			current_pair = to_solve[:, m]
			# Generate matrix of indices which fill the box defined by the origin and our current point
			# Find pairs of vectors which span the box and sum to the current vector
			A = np.indices((current_pair[0] + 1, current_pair[1] + 1))
			B = np.indices((current_pair[0] + 1, current_pair[1] + 1))
			B[0, :, :] = current_pair[0] - B[0, :, :]
			B[1, :, :] = current_pair[1] - B[1, :, :]
			# Flatten in to list of pairs and remove trivial (0,0) + (n,m) pairs
			A = A.reshape((2, -1))
			B = B.reshape((2, -1))
			A = A[:, 1:-1]
			B = B[:, 1:-1]

			plus = np.empty((len(A[0, :])))
			minus = np.empty((len(A[0, :])))
			for i in range(len(A[0, :])):
				# Find the positive and negative solutions
				plus[i] = Phi[A[0, i], A[1, i], B[0, i], B[1, i]] + solved[
					A[0, i], A[1, i]] + solved[B[0, i], B[1, i]]
				minus[i] = -Phi[A[0, i], A[1, i], B[0, i], B[1, i]] + solved[
					A[0, i], A[1, i]] + solved[B[0, i], B[1, i]]

			theta1 = np.append(plus, minus)
			theta2 = np.append(minus, plus)

			xdata = np.cos(theta1)
			ydata = np.sin(theta2)

			print(current_pair)
			# If an alternate has been requested by the user for the pixel, choose the other value
			if Alt[current_pair[0], current_pair[1]] == 1:
				next_phi, error_val = self.find_next_phi(xdata=xdata,
														 ydata=ydata,
														 AltReturn=True)
			else:
				next_phi, error_val = self.find_next_phi(xdata=xdata,
														 ydata=ydata)

			solved[current_pair[0], current_pair[1]] = next_phi
			error[current_pair[0], current_pair[1]] = error_val
		n += 1

	# Solve phase out to q_max, at this point error resolving should not be needed
	for n in range(1, len(Phi[0, 0, 0, :])):
		# Generate list of points across the diagonal to be solved this round
		to_solve_1 = np.arange(len(Phi[0, 0, 0, :]) - n) + n
		to_solve_2 = to_solve_1[::-1]

		to_solve = np.asarray([to_solve_1, to_solve_2])

		for m in range(len(to_solve[0, :])):
			current_pair = to_solve[:, m]
			# Generate matrix of indices which fill the box defined by the origin and our current point
			# Find pairs of vectors which span the box and sum to the current vector
			# current_pair[np.argmin(current_pair)] += 1
			# current_pair[np.argmax(current_pair)] -=1
			# A = np.indices(current_pair)
			# B = np.indices(current_pair)
			A = np.mgrid[0:current_pair[0] + 1, 0:current_pair[1] + 1]
			B = np.mgrid[0:current_pair[0] + 1, 0:current_pair[1] + 1]
			B[0, :, :] = current_pair[0] - B[0, :, :]
			B[1, :, :] = current_pair[1] - B[1, :, :]
			# Flatten in to list of pairs and remove trivial (0,0) + (n,m) pairs
			A = A.reshape((2, -1))
			B = B.reshape((2, -1))
			A = A[:, 1:-1]
			B = B[:, 1:-1]

			plus = np.empty((len(A[0, :])))
			minus = np.empty((len(A[0, :])))
			for i in range(len(A[0, :])):
				# Find the positive and negative solutions
				plus[i] = Phi[A[0, i], A[1, i], B[0, i], B[1, i]] + solved[
					A[0, i], A[1, i]] + solved[B[0, i], B[1, i]]
				minus[i] = -Phi[A[0, i], A[1, i], B[0, i], B[1, i]] + solved[
					A[0, i], A[1, i]] + solved[B[0, i], B[1, i]]

			theta1 = np.append(plus, minus)
			theta2 = np.append(minus, plus)

			xdata = np.cos(theta1)
			ydata = np.sin(theta2)

			print(current_pair)
			next_phi, error_val = self.find_next_phi(xdata=xdata, ydata=ydata)

			solved[current_pair[0], current_pair[1]] = next_phi
			error[current_pair[0], current_pair[1]] = error_val

	return solved, error


def find_next_phi(self, xdata=None, ydata=None, AltReturn=False):
	"""Form a complex number.

	Keyword arguments:
	real -- the real part (default 0.0)
	imag -- the imaginary part (default 0.0)
	"""

	# Samples the error function and starts minimization near the minimum

	def thetaError(theta):
		return np.minimum((np.add.outer(xdata, -np.cos(theta)))**2,
						  (np.add.outer(ydata, -np.sin(theta)))**2).sum(0)

	def logThetaError(theta):
		return np.log(np.minimum((np.add.outer(xdata, -np.cos(theta)))**2,
								 (np.add.outer(ydata, -np.sin(theta)))**2).sum(
			0))

	def ABError(AB):
		return np.log(np.minimum((np.add.outer(xdata, -AB[0, :, :]))**2,
								 (np.add.outer(ydata, -AB[1, :, :]))**2).sum(0))

	def opt_func(theta):
		if np.abs(theta) > np.pi:
			return 1e10
		else:
			return np.log(np.sum(np.minimum((xdata - np.cos(theta))**2,
											(ydata - np.sin(theta))**2)))

	# This error function has negative poles at the solution
	# Search for points theta that have a very large second derivative to find the poles
	theta = np.linspace(-np.pi, np.pi, 50000)
	thetaError = thetaError(theta)
	logThetaError = logThetaError(theta)
	dthetaError = np.gradient(logThetaError, theta)
	ddthetaError = np.gradient(dthetaError, theta)
	num_theta = 2  # Number of candidates to accept. Two is optimal.
	mask = (np.argpartition(ddthetaError, -num_theta)[
			-num_theta:])  # Indices where second derivative is maximal

	# Why not just brute force calculate the minimum of the error function?
	# Why was calculating the second derivative necessary?
	# mask = (np.argpartition(logThetaError, num_theta)[:num_theta])
	print("Possible Theta = ", theta[mask])
	theta0 = theta[mask]

	# Optimize candidate theta and choose the theta with smallest error
	fCandidate = []
	thetaCandidate = []
	for val in theta0:
		res = optimize.minimize(opt_func, x0=val, method='CG', tol=1e-10,
								options={'gtol': 1e-8, 'maxiter': 10000})
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