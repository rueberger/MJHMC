"""
This file contains sampling algorithms that work over a finite state ladder
  with specified energies, and is hence referred to as algebraic_hmc.

These implementations are used to compute and validate the analytic spectral
  gap result from the paper.

They are of no use for practical sampling purposes. Please see the core MJHMC
 implementation in markov_jump_hmc.py
"""

import numpy as np
import copy
import itertools
from mjhmc.misc.utils import overrides, min_idx, normalize_by_row

#pylint: disable=too-many-instance-attributes

class AlgebraicDiscrete(object):
    """
    Implements an algebraic hmc-like sampler (although its not quite accurate to call it hmc - more like
    dihedral monte carlo)
    That is, explores a single state ladder with random energies and not a real distribution
    """

    def __init__(self, order,
                 energies=None,
                 batch_size=100):
        """
        order is the number of elements in the state group,
           twice the number of energy levels
        energies real valued array, otherwise initialized randomly
        """
        self.order = order
        self.nbatch = batch_size
        self.ladder_states = State(self.order, self.nbatch, energies)
        self.transitions = np.zeros((self.order, self.order))
        self.distr = np.zeros(self.order / 2)
        self.prd_distr = self.calculate_true_distribution()
        self.mix_eps = .01
        # number of sampling steps total
        self.n = 0
        # steps to take before taking samples
        self.burn_in_steps = 1000
        # probability to flip momentum
        self.p_flip = .5

        # auxiliary state ladder group used to calculate transition matrix
        self.aux_ladder = StateGroup(self.order, np.ones(self.order / 2))
        self.idx_to_state = self.aux_ladder.get_idx_map()

    def sampling_iteration(self):
        """Perform a single sampling step
        """
        # states
        pre_state = self.ladder_states.copy()
        fl_state = self.ladder_states.copy().FL()

        # rates
        p_fl = self.acceptance_rate(self.ladder_states, fl_state)
        p_flip = self.p_flip * np.ones(self.nbatch)

        # accepted transitions
        fl_acc_idx = np.arange(self.nbatch)[np.random.rand(self.nbatch) < p_fl]
        f_idx = np.arange(self.nbatch)[np.random.rand(self.nbatch) < p_flip]

        # update state
        self.ladder_states.update(fl_acc_idx, fl_state)
        self.ladder_states.update(f_idx, self.ladder_states.copy().F())
        self.n += 1
        if self.n > self.burn_in_steps:
            self.update_empirical_transition_matrix(pre_state)
            self.update_empirical_distr()

    def sample(self, iterations=1000, burn_in=False):
        """Calls sampling_iteration iterations times
        """
        if burn_in and self.n < self.burn_in_steps:
            for _ in xrange(self.burn_in_steps - self.n):
                self.sampling_iteration()
        for _ in xrange(iterations):
            self.sampling_iteration()

    def burn_in(self):
        """Runs the sampler until burn in
        """
        self.sample(self.burn_in_steps)

    def acceptance_rate(self, Z1, Z2):
        """Metropolis-Hasting transition probability from Z1 to Z2
        note that Z1 and Z2 are states objects
        """
        EZ1 = Z1.H()
        EZ2 = Z2.H()
        Ediff = EZ1 - EZ2
        p_acc = np.ones(self.nbatch)
        p_acc[Ediff < 0] = np.exp(Ediff[Ediff < 0])
        return p_acc

    def idx_acceptance_rate(self, pre_idx, post_idx):
        """Returns the acceptance rate for transitions
        from state pre_idx to post_idx
        """
        pre_E = self.ladder_states.energies[pre_idx % (self.order / 2)]
        post_E = self.ladder_states.energies[post_idx % (self.order / 2)]
        Ediff = pre_E - post_E
        return min(1, np.exp(Ediff))

    def update_empirical_transition_matrix(self, pre_state):
        """updates the transition matrix
        """
        for pre_idx, curr_idx in zip(pre_state.full_idxs(),
                                     self.ladder_states.full_idxs()):
            self.transitions[pre_idx, curr_idx] += 1

    def update_empirical_distr(self):
        """updates the recorded distribution
        """
        for idx in self.ladder_states.idxs():
            self.distr[idx] += 1

    def get_empirical_distr(self):
        """returns the normalized distribution
        """
        if self.n < self.burn_in_steps:
            raise RuntimeWarning("Not burned in yet")
        norm = np.sum(self.distr)
        return self.distr / norm

    def get_empirical_transition_matrix(self, full=False):
        """Normalizes and returns the transition matrix
        """
        if self.n < self.burn_in_steps:
            raise RuntimeWarning("Not burned in yet")
        if full:
            return normalize_by_row(self.transitions)
        else:
            reduced = self.reduce_full_transition_matrix(self.transitions)
            return normalize_by_row(reduced)

    def calculate_true_transition_matrix(self, full=True):
        """
        Returns the analytic transition matrix for this distribution
        Uses transition probabilities to build matrix for full state space (flipped states too)
        and then sums out flipped states and returns the matrix for energy states only
        """
        # no nice way to have private variables...
        T_full = np.zeros((self.order, self.order))

        for idx in xrange(self.order):
            self.update_transition_matrix(idx, T_full)

        if full:
            return normalize_by_row(T_full)
        else:
            T_l = self.reduce_full_transition_matrix(T_full)
            return normalize_by_row(T_l)

    def reduce_full_transition_matrix(self, T_full):
        """
        T_full an array with size (order, order)
        sums over T_full to return the equivalent transition matrix over just energy states
        returns an array with size (order / 2, order / 2)
        """
        n_energies = self.order / 2

        def get_eq_idxs(l_i, l_j):
            """
            Returns the four pairs of indices equivalent to l_i, l_j in the full space
            returns [(l_i, l_j), (l_i, f_j), (f_i, l_j )]
            """
            state_l_i = self.idx_to_state[l_i]
            state_l_j = self.idx_to_state[l_j]
            state_f_i = (1, state_l_i[1])
            state_f_j = (1, state_l_j[1])
            f_i = self.aux_ladder.idx_of(state_f_i)
            f_j = self.aux_ladder.idx_of(state_f_j)
            return list(itertools.product([l_i, f_i], [l_j, f_j]))

        T_l = np.zeros((n_energies, n_energies))

        for l_i, l_j in itertools.product(
                np.arange(n_energies), np.arange(n_energies)):
            T_l[l_i, l_j] = np.sum(T_full[zip(*get_eq_idxs(l_i, l_j))])

        return T_l

    def update_transition_matrix(self, idx, T_full):
        """
        Calculates transition probabilties to states reachable from idx
        updates the transition matrix
        """
        state = self.idx_to_state[idx]
        fl_idx = self.aux_ladder.fl_idx_of(state)
        f_idx = self.aux_ladder.f_idx_of(state)
        l_idx = self.aux_ladder.l_idx_of(state)
        p_fl = self.idx_acceptance_rate(idx, fl_idx)
        # accounting for non "commutativity" of MH rate
        p_stay = 1 - p_fl
        # flip momentum with probability self.p_flip
        p_transitions = ([p_stay * (1 - self.p_flip), p_fl * (1 - self.p_flip)] +
                         [p_stay * self.p_flip, p_fl * self.p_flip])
        transition_idxs = [idx, fl_idx, f_idx, l_idx]
        for t_idx, p_t in zip(transition_idxs, p_transitions):
            T_full[idx, t_idx] = p_t

    def calculate_true_distribution(self):
        """Returns the distribution predicted by theory
        """
        p = np.zeros(self.order / 2)
        for idx in xrange(self.order / 2):
            p[idx] = np.exp(-self.ladder_states.energies[idx])
        return p / np.sum(p)

    def calculate_mixing_time(self):
        """Runs the sampler until our sampled distribution is within
        self.mix_eps of the true distribution
        Returns the number of sampling steps needed
        """
        assert self.n == 0
        # avoids divide by 0 for normalizing sampled distr
        self.burn_in()
        self.sampling_iteration()
        while np.sum(np.abs(self.get_empirical_distr() - self.prd_distr)) >= self.mix_eps:
            self.sampling_iteration()
        return self.n

class AlgebraicHMC(AlgebraicDiscrete):
    """
    implements AlgebraicHMC which is accomplished just by cranking up self.p_flip of
    AlgebraicDiscrete to 1
    """

    def __init__(self, *args, **kwargs):
        super(AlgebraicHMC, self).__init__(*args, **kwargs)
        self.p_flip = 1


class AlgebraicContinuous(AlgebraicDiscrete):
    """The continuous analog of above
    """

    def __init__(self, order,
                 energies=None,
                 batch_size=100):

        super(AlgebraicContinuous, self).__init__(
            order, energies, batch_size)
        # rate of transitions into the flipped state
        self.f_rate = 1

    @overrides(AlgebraicDiscrete)
    def acceptance_rate(self, Z1, Z2):
        """
        transition rate from Z1 to Z2
        rate = [p(Z1)/p(Z2)]^1/2
        """
        E1 = Z1.H()
        E2 = Z2.H()
        Ediff = E1 - E2
        return np.exp(Ediff * .5)

    @overrides(AlgebraicDiscrete)
    def idx_acceptance_rate(self, pre_idx, post_idx):
        """Returns the acceptance rate for transitions
        from state pre_idx to post_idx
        """
        pre_E = self.ladder_states.energies[pre_idx % (self.order / 2)]
        post_E = self.ladder_states.energies[post_idx % (self.order / 2)]
        Ediff = pre_E - post_E
        # I think this is wrong!!!!!
        return np.exp(Ediff * .5)

    @overrides(AlgebraicDiscrete)
    def sampling_iteration(self):
        """perform a single sampling step
        """
        # states
        pre_state = self.ladder_states.copy()
        f_state = self.ladder_states.copy().F()
        fl_state = self.ladder_states.copy().FL()

        # rates
        fl_rates = self.acceptance_rate(self.ladder_states, fl_state)
        f_rates = self.f_rate * np.ones(self.nbatch)

        # draws from exponential distribution
        fl_draws = self.draw_from(fl_rates)
        f_draws = self.draw_from(f_rates)

        # select minimum waiting times
        fl_idx, f_idx = min_idx([fl_draws, f_draws])
        waiting_times = np.minimum(f_draws, fl_draws)[0]

        self.n += 1

        # update last states with waiting times
        if self.n > self.burn_in_steps:
            self.update_distr(waiting_times)

        self.ladder_states.update(fl_idx, fl_state)
        self.ladder_states.update(f_idx, f_state)

        if self.n > self.burn_in_steps:
            self.update_empirical_transition_matrix(pre_state)

    def draw_from(self, rates):
        """
        returns an array of draws from an exponential distribution with rate
        """
        return np.array([np.random.exponential(scale=1./rate)
                         if rate != 0 else np.inf
                         for rate in rates]).reshape(1, self.nbatch)

    def update_distr(self, waiting_times):
        """updates the distribution with waiting times
        """
        for idx, wait in zip(self.ladder_states.idxs(), waiting_times):
            self.distr[idx] += wait

    @overrides(AlgebraicDiscrete)
    def update_transition_matrix(self, idx, T_full):
        """
        Calculates transition probabilties to states reachable from idx
        updates the transition matrix
        """
        state = self.idx_to_state[idx]
        fl_idx = self.aux_ladder.fl_idx_of(state)
        f_idx = self.aux_ladder.f_idx_of(state)
        fl_rate = self.idx_acceptance_rate(idx, fl_idx)
        p_fl = fl_rate / (fl_rate + self.f_rate)
        p_f = 1 - p_fl

        p_transitions = [p_fl, p_f]
        transition_idxs = [fl_idx, f_idx]
        for t_idx, p_t in zip(transition_idxs, p_transitions):
            T_full[idx, t_idx] = p_t

class AlgebraicReducedFlip(AlgebraicContinuous):
    """The continuous algebraic sampler with reduced flip rate
    """

    @overrides(AlgebraicContinuous)
    def sampling_iteration(self):
        # states
        pre_state = self.ladder_states.copy()
        f_state = self.ladder_states.copy().F()
        l_state = self.ladder_states.copy().F().FL()
        # aka l^-1 state
        flf_state = self.ladder_states.copy().FL().F()

        # rates
        l_rates = self.acceptance_rate(self.ladder_states, l_state)
        flf_rates = self.acceptance_rate(self.ladder_states, flf_state)
        f_rates = flf_rates - np.min((flf_rates, l_rates), axis=0)

        # draws from exponential distribution
        l_draws = self.draw_from(l_rates)
        f_draws = self.draw_from(f_rates)

        # select minimum waiting times
        l_idx, f_idx = min_idx([l_draws, f_draws])
        waiting_times = np.minimum(f_draws, l_draws)[0]

        self.n += 1

        # update last states with waiting times
        if self.n > self.burn_in_steps:
            self.update_distr(waiting_times)

        self.ladder_states.update(l_idx, l_state)
        self.ladder_states.update(f_idx, f_state)

        if self.n > self.burn_in_steps:
            self.update_empirical_transition_matrix(pre_state)


    @overrides(AlgebraicContinuous)
    def update_transition_matrix(self, idx, T_full):
        """
        Calculates transition probabilties to states reachable from idx
        updates the transition matrix
        """
        state = self.idx_to_state[idx]
        l_idx = self.aux_ladder.l_idx_of(state)
        flf_idx = self.aux_ladder.flf_idx_of(state)
        f_idx = self.aux_ladder.f_idx_of(state)

        l_rate = self.idx_acceptance_rate(idx, l_idx)
        flf_rate = self.idx_acceptance_rate(idx, flf_idx)
        f_rate = flf_rate - min(l_rate, flf_rate)

        p_l = l_rate / (l_rate + f_rate)
        p_f = 1 - p_l

        p_transitions = [p_l, p_f]
        transition_idxs = [l_idx, f_idx]
        for t_idx, p_t in zip(transition_idxs, p_transitions):
            T_full[idx, t_idx] = p_t


class State(object):
    """Simple wrapper object for holding state
    """
    def __init__(self, order, nbatch, energies=None):
        self.order = order
        self.nbatch = nbatch
        # normally distribution energies
        # draw from different distribution?
        if energies is not None:
            self.energies = energies
        else:
            self.energies = np.random.randn(self.order / 2)
        self.states = np.array([StateGroup(self.order, self.energies) for _ in xrange(self.nbatch)])

    def H(self):
        """Returns an array with the energies of this state
        """
        return np.array([ladder.E() for ladder in self.states])

    def FL(self):
        """applies FL to all of the ladders
        returns self for convenience
        """
        for ladder in self.states:
            ladder.FL()
        return self

    def F(self):
        """applies F to all of the ladders
        returns self for convenience
        """
        for ladder in self.states:
            ladder.F()
        return self


    def copy(self):
        """returns a deep copy of this object
        """
        # really should just use deep copy method
        state_copy = State(self.order, self.nbatch, self.energies)
        for curr_lad, copy_lad in zip(self.states, state_copy.states):
            copy_lad.state = copy.copy(curr_lad.state)
        return state_copy

    def update(self, idx, Z):
        """replace batch elements idx with state from Z
        """
        for i in idx:
            # awful, i know
            self.states[i].state = Z.states[i].state

    def idxs(self):
        """return an array of the indices of the current ladder states
        """
        return np.array([ladder.idx() for ladder in self.states])

    def full_idxs(self):
        """
        return array of the full indices
        """
        return np.array([ladder.full_idx() for ladder in self.states])


class StateGroup(object):
    """Holds ladder state and implements group operation
    Isomorphic to the dihedral group of order 2n with L^n = 1, F^2 = 1,
    This implementation uses that fact that any element h of D_2n can be
       written as h = F^k_1 L^k_2 where k_1 = 0,1 and k_2 = 0,1...n-1
    Not user facing
    """

    def __init__(self, order, energies):
        # order must be even
        assert order % 2 == 0
        assert len(energies) == order / 2

        self.order = order
        # [k_1, k_2]
        self.state = [np.random.randint(0, 2), np.random.randint(0, order / 2)]
        self.energies = energies


    def FL(self):
        """composes FL with the current state
        """
        if self.state[0] == 1:
            self.state[0] = 0
            self.state[1] = (self.state[1] - 1) % (self.order / 2)
        else:
            self.state[0] = 1
            self.state[1] = (self.state[1] + 1) % (self.order / 2)

    def F(self):
        """
        Composes F with the current state
        """
        self.state[0] = (self.state[0] + 1) % 2

    def L(self):
        """
        Composes L with the current state
        """
        self.FL()
        self.F()


    def idx(self):
        """returns the current state (not from the extended distribution)
        """
        return self.state[1]

    def full_idx(self):
        """
        bijective map of current state onto the integers 0 - 2(n-1) for calculating the transition matrix
        difference between this and the self.idx is that it accounts for flipped states
        """
        return (self.order / 2) * self.state[0] + self.state[1]

    def idx_to_kp(self, idx):
        """ Compute the (p, k) pair corresponding to idx
        """
        midpt = (self.order / 2)
        if idx >= midpt:
            return (1, idx - midpt)
        else:
            return (0, idx)



    def idx_of(self, state):
        """
        wrapper function that returns full idx of state
        mutates state
        """
        self.state = state
        return self.full_idx()

    def fl_idx_of(self, state):
        """
        returns full idx of FL state
        mutates state
        """
        self.state = list(state)
        self.FL()
        return self.full_idx()

    def f_idx_of(self, state):
        """
        returns full idx of F state
        mutates state
        """
        self.state = list(state)
        self.F()
        return self.full_idx()

    def flf_idx_of(self, state):
        """
        returns full idx of FLF state
        mutates state
        """
        self.state = list(state)
        self.F()
        self.FL()
        return self.full_idx()

    def l_idx_of(self, state):
        """
        returns full idx of L state
        mutates state
        """
        self.state = list(state)
        self.FL()
        self.F()
        return self.full_idx()

    def E(self):
        """Returns the energy of the current state
        """
        return self.energies[self.state[1]]

    def get_idx_map(self):
        """
        returns a dictionary from index to the state mapping to it under full_idx
        """
        return {self.idx_of(state): state for state in itertools.product(
            np.arange(2),
            np.arange(self.order / 2))}
