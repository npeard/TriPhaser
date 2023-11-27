#!/usr/bin/env python

import numpy as np
import pylab as P
import ase

class Xtal:
    def __init__(self, superLattice=(2,2,2)):
        import ase.lattice.cubic as ALC
        import ase.lattice.hexagonal as ALH
        #self.atoms = ALC.FaceCenteredCubic(size=superLattice, symbol='Cu', pbc=(1,1,1))
        self.atoms = ALH.HexagonalClosedPacked(size=superLattice, symbol='Cu', pbc=(1,1,1), latticeconstant=(2.,2))
        # Convert positions to meters
        self.atoms.set_cell(np.identity(3)*self.atoms.get_cell())


    def get_positions(self):
        # Units in meters
        return self.atoms.get_positions()


    def get_number_atoms(self):
        return len(self.atoms)