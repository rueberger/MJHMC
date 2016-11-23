"""
This file contains the class HMC state which holds state variables,
  provides leapfrog integration methods and serves as a cache
"""

import numpy as np

#pylint: disable=too-many-class-attributes

class HMCState(object):
    """ Holds all the state variables for sampling particles."""

    def __init__(self, X, parent, V=None, EX=None, EV=None, dEdX=None, slave=False):
        """
        Initialize sampling particle states.  Called by all continuous state
         space sampler classes
        Not user facing.
        """
        self.parent = parent
        self.X = X
        self.V = V
        self.nbatch = X.shape[1]
        self.active_idx = np.arange(self.nbatch)
        if V is None:
            N = self.X.shape[0]
            self.V = np.random.randn(N, self.nbatch)
        self.EX = EX
        if EX is None:
            self.EX = np.zeros((1,self.nbatch))
            self.update_EX()

        self.EV = EV
        if EV is None:
            self.EV = np.zeros((1,self.nbatch))
            self.update_EV()
        self.dEdX = dEdX
        if dEdX is None:
            self.dEdX = np.zeros(X.shape)
            self.update_dEdX()

        if not slave:
            self.cached_flf_state = self.copy(copy_slave=True)
            # True at idx if the cache holds the flf_state at idx
            self.cache_active = np.array([False] * self.nbatch)

    def update_EX(self):
        self.EX[:,self.active_idx] = self.parent.E(self.X[:,self.active_idx]).reshape((1,-1))

    def update_EV(self):
        self.EV[:,self.active_idx] = np.sum(self.V[:,self.active_idx]**2, axis=0).reshape((1,-1))/2.

    def update_dEdX(self):
        self.dEdX[:,self.active_idx] = self.parent.dEdX(self.X[:,self.active_idx])

    def copy(self, copy_slave=False):
        Z = HMCState(self.X.copy(), self.parent, V=self.V.copy(), EX=self.EX.copy(), EV=self.EV.copy(), dEdX=self.dEdX.copy(), slave=copy_slave)
        Z.active_idx = self.active_idx.copy()
        if not copy_slave:
            Z.cached_flf_state = self.cached_flf_state.copy(True)
            Z.cache_active = self.cache_active.copy()
        return Z

    def update(self, idx, Z):
        """ replace batch elements idx with state from Z """
        # may be able to remove this
        if len(idx) == 0:
            return
        self.X[:, idx] = Z.X[:, idx]
        self.V[:, idx] = Z.V[:, idx]
        self.EX[:, idx] = Z.EX[:, idx]
        self.EV[:, idx] = Z.EV[:, idx]
        self.dEdX[:, idx] = Z.dEdX[:, idx]

    def get_state(self):
        """returns the concatentaion of X and V
        For use in eigs.py. Don't use this if nbatch > 1
        """
        return np.concatenate((self.X, self.V))

    def H(self):
        """
        returns the full energy of the state
        """
        return self.EX + self.EV

    def leapfrog(self):
        """ A single leapfrog step for X and V """
        self.V[:, self.active_idx] += -self.parent.epsilon/2. * self.dEdX[:, self.active_idx]
        self.X[:, self.active_idx] += self.parent.epsilon * self.V[:, self.active_idx]
        self.update_dEdX()
        self.V[:, self.active_idx] += -self.parent.epsilon/2. * self.dEdX[:, self.active_idx]

    def L(self):
        """ Run the leapfrog operator for M leapfrog steps
        returns self for convenience"""
        for _ in range(self.parent.num_leapfrog_steps):
            self.leapfrog()
        self.update_EV()
        self.update_EX()
        return self

    def F(self):
        """Explicity flip operator for readability
        returns self for convenience
        """
        self.V[:, self.active_idx] = - self.V[:, self.active_idx]
        return self

    def FLF(self):
        """
        Returns the FLF state
        reads from the cache if possible
        """
        cached_idx = np.where(self.cache_active == True)[0]
        self.active_idx = np.where(self.cache_active == False)[0]
        flf_state = self.F().L().F()
        flf_state.update(cached_idx, self.cached_flf_state)
        self.active_idx = np.arange(self.nbatch)
        return flf_state

    def R(self):
        """randomizes the momentum with rate beta
        resets the cache
        return self for convenience
        """
        self.V = self.V*np.sqrt(1.-self.parent.beta) + np.random.randn(
            self.parent.ndims, self.nbatch)*np.sqrt(self.parent.beta)
        self.update_EV()
        return self

    def cache_flf_state(self, idx, Z):
        """
        stores a copy of Z as the flf_state
        """
        self.cached_flf_state.update(idx, Z)
        self.cache_active[idx] = True

    def clear_flf_cache(self, idx):
        """
        clears the cached flf state
        """
        # to be called from update
        self.cache_active[idx] = False

    def reset_flf_cache(self):
        """ Entirely wipes the cache
        """
        self.cache_active = self.zeros_like(self.cache_active)
