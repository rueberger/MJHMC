import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict

def simulate_dynamics(initial_pos, initial_vel, stepsize, n_steps, energy_fn):
    """
    Return final (position, velocity) obtained after an `n_steps` leapfrog
    updates, using Hamiltonian dynamics.

    Parameters
    ----------
    initial_pos: shared theano matrix
        Initial position at which to start the simulation
    initial_vel: shared theano matrix
        Initial velocity of particles
    stepsize: shared theano scalar
        Scalar value controlling amount by which to move
    n_steps: shared theano scalar 
        Scalar value controlling number of steps for which to run the integrator
    energy_fn: python function
        Python function, operating on symbolic theano variables, used to
        compute the potential energy at a given position.

    Returns
    -------
    rval1: theano matrix
        Final positions obtained after simulation
    rval2: theano matrix
        Final velocity obtained after simulation
    """

    def leapfrog(pos, vel, step):
        """
        Inside loop of Scan. Performs one step of leapfrog update, using
        Hamiltonian dynamics.

        Parameters
        ----------
        pos: theano matrix
            in leapfrog update equations, represents pos(t), position at time t
        vel: theano matrix
            in leapfrog update equations, represents vel(t - stepsize/2),
            velocity at time (t - stepsize/2)
        step: theano scalar
            scalar value controlling amount by which to move

        Returns
        -------
        rval1: [theano matrix, theano matrix]
            Symbolic theano matrices for new position pos(t + stepsize), and
            velocity vel(t + stepsize/2)
        rval2: dictionary
            Dictionary of updates for the Scan Op
        """
        # from pos(t) and vel(t-stepsize/2), compute vel(t+stepsize/2)
        dE_dpos = T.grad(energy_fn(pos).sum(), pos)
        new_vel = vel - step * dE_dpos
        # from vel(t+stepsize/2) compute pos(t+stepsize)
        new_pos = pos + step * new_vel
        return [new_pos, new_vel], {}

    # compute velocity at time-step: t + stepsize/2
    initial_energy = energy_fn(initial_pos)
    dE_dpos = T.grad(initial_energy.sum(), initial_pos)
    vel_half_step = initial_vel - 0.5 * stepsize * dE_dpos

    # compute position at time-step: t + stepsize
    pos_full_step = initial_pos + stepsize * vel_half_step

    # perform leapfrog updates: the scan op is used to repeatedly compute
    # vel(t + (m-1/2)*stepsize) and pos(t + m*stepsize) for m in [2,n_steps].
    (all_pos, all_vel), scan_updates = theano.scan(
        leapfrog,
        outputs_info=[
            dict(initial=pos_full_step),
            dict(initial=vel_half_step),
        ],
        non_sequences=[stepsize],
        n_steps=n_steps - 1)
    final_pos = all_pos[-1]
    final_vel = all_vel[-1]
    # NOTE: Scan always returns an updates dictionary, in case the
    # scanned function draws samples from a RandomStream. These
    # updates must then be used when compiling the Theano function, to
    # avoid drawing the same random numbers each time the function is
    # called. In this case however, we consciously ignore
    # "scan_updates" because we know it is empty.
    assert not scan_updates

    # The last velocity returned by scan is vel(t +
    # (n_steps - 1 / 2) * stepsize) We therefore perform one more half-step
    # to return vel(t + n_steps * stepsize)
    energy = energy_fn(final_pos)
    final_vel = final_vel - 0.5 * stepsize * T.grad(energy.sum(), final_pos)

    # return new proposal state
    return final_pos, final_vel


# start-snippet-1


def kinetic_energy(vel):
    """Returns the kinetic energy associated with the given velocity
    and mass of 1.

    Parameters
    ----------
    vel: theano matrix
        Symbolic matrix whose rows are velocity vectors.

    Returns
    -------
    return: theano vector
        Vector whose i-th entry is the kinetic entry associated with vel[i].

    """
    return 0.5 * (vel ** 2).sum(axis=0)

def hamiltonian(pos, vel, energy_fn):
    """
    Returns the Hamiltonian (sum of potential and kinetic energy) for the given
    velocity and position.

    Parameters
    ----------
    pos: theano matrix
        Symbolic matrix whose rows are position vectors.
    vel: theano matrix
        Symbolic matrix whose rows are velocity vectors.
    energy_fn: python function
        Python function, operating on symbolic theano variables, used tox
        compute the potential energy at a given position.

    Returns
    -------
    return: theano vector
        Vector whose i-th entry is the Hamiltonian at position pos[i] and
        velocity vel[i].
    """
    # assuming mass is 1
    return energy_fn(pos) + kinetic_energy(vel)

def metropolis_hastings_accept(energy_prev, energy_next, s_rng):
    """
    Performs a Metropolis-Hastings accept-reject move.

    Parameters
    ----------
    energy_prev: theano vector
        Symbolic theano tensor which contains the energy associated with the
        configuration at time-step t.
    energy_next: theano vector
        Symbolic theano tensor which contains the energy associated with the
        proposed configuration at time-step t+1.
    s_rng: theano.tensor.shared_randomstreams.RandomStreams
        Theano shared random stream object used to generate the random number
        used in proposal.

    Returns
    -------
    return: boolean
        True if move is accepted, False otherwise
    """
    ediff = energy_prev - energy_next
    return (T.exp(ediff) - s_rng.uniform(size=energy_prev.shape)) >= 0

def MJHMC_accept():

    return

def hmc_move(s_rng, positions, energy_fn, stepsize=0.1, n_steps=1):
    """
    This function performs one-step of Hybrid Monte-Carlo sampling. We start by
    sampling a random velocity from a univariate Gaussian distribution, perform
    `n_steps` leap-frog updates using Hamiltonian dynamics and accept-reject
    using Metropolis-Hastings.

    Parameters
    ----------
    s_rng: theano shared random stream
        Symbolic random number generator used to draw random velocity and
        perform accept-reject move.
    positions: shared theano matrix
        Symbolic matrix whose rows are position vectors.
    energy_fn: python function
        Python function, operating on symbolic theano variables, used to
        compute the potential energy at a given position.
    stepsize:  shared theano scalar
        Shared variable containing the stepsize to use for `n_steps` of HMC
        simulation steps.
    n_steps: integer
        Number of HMC steps to perform before proposing a new position.

    Returns
    -------
    rval1: boolean
        True if move is accepted, False otherwise
    rval2: theano matrix
        Matrix whose rows contain the proposed "new position"
    """
    # sample random velocity
    initial_vel = s_rng.normal(size=positions.shape)

    final_pos, final_vel = simulate_dynamics(
        initial_pos=positions,
        initial_vel=initial_vel,
        stepsize=stepsize,
        n_steps=n_steps,
        energy_fn=energy_fn
    )
    # accept/reject the proposed move based on the joint distribution
    accept = metropolis_hastings_accept(
        energy_prev=hamiltonian(positions, initial_vel, energy_fn),
        energy_next=hamiltonian(final_pos, final_vel, energy_fn),
        s_rng=s_rng
    )
    return accept, final_pos


def hmc_updates(positions,final_pos, accept):
    """def hmc_updates(positions, stepsize, avg_acceptance_rate, final_pos, accept,
                target_acceptance_rate, stepsize_inc, stepsize_dec,
                stepsize_min, stepsize_max, avg_acceptance_slowness):
    This function is executed after `n_steps` of HMC sampling
    (`hmc_move` function). It creates the updates dictionary used by
    the `simulate` function. It takes care of updating: the position
    (if the move is accepted), the stepsize (to track a given target
    acceptance rate) and the average acceptance rate (computed as a
    moving average).

    Parameters
    ----------
    positions: shared variable, theano matrix
        Shared theano matrix whose rows contain the old position
    stepsize: shared variable, theano scalar
        Shared theano scalar containing current step size
    avg_acceptance_rate: shared variable, theano scalar
        Shared theano scalar containing the current average acceptance rate
    final_pos: shared variable, theano matrix
        Shared theano matrix whose rows contain the new position
    accept: theano scalar
        Boolean-type variable representing whether or not the proposed HMC move
        should be accepted or not.
    target_acceptance_rate: float
        The stepsize is modified in order to track this target acceptance rate.
    stepsize_inc: float
        Amount by which to increment stepsize when acceptance rate is too high.
    stepsize_dec: float
        Amount by which to decrement stepsize when acceptance rate is too low.
    stepsize_min: float
        Lower-bound on `stepsize`.
    stepsize_min: float
        Upper-bound on `stepsize`.
    avg_acceptance_slowness: float
        Average acceptance rate is computed as an exponential moving average.
        (1-avg_acceptance_slowness) is the weight given to the newest
        observation.

    Returns
    -------
    rval1: dictionary-like
        A dictionary of updates to be used by the `HMC_Sampler.simulate`
        function.  The updates target the position, stepsize and average
        acceptance rate.

    """

    ## POSITION UPDATES ##
    # broadcast `accept` scalar to tensor with the same dimensions as
    # final_pos.
    #accept_matrix = accept.dimshuffle(0, *(('x',) * (final_pos.ndim - 1)))
    # if accept is True, update to `final_pos` else stay put
    #new_positions = T.switch(accept_matrix, final_pos, positions)
    #new_positions = T.switch(accept.ravel(), final_pos, positions).astype('float32')
    #new_positions = T.switch(accept[0].dimshuffle('x',0), final_pos, positions).astype('float32')
    new_positions = accept[0]*final_pos + (1-accept[0])*positions
    ## ACCEPT RATE UPDATES ##
    # perform exponential moving average
    '''
    mean_dtype = theano.scalar.upcast(accept.dtype, avg_acceptance_rate.dtype)
    new_acceptance_rate = T.add(
        avg_acceptance_slowness * avg_acceptance_rate,
        (1.0 - avg_acceptance_slowness) * accept.mean(dtype=mean_dtype))
    ## STEPSIZE UPDATES ##
    # if acceptance rate is too low, our sampler is too "noisy" and we reduce
    # the stepsize. If it is too high, our sampler is too conservative, we can
    # get away with a larger stepsize (resulting in better mixing).
    _new_stepsize = T.switch(avg_acceptance_rate > target_acceptance_rate,
                              stepsize * stepsize_inc, stepsize * stepsize_dec)
    # maintain stepsize in [stepsize_min, stepsize_max]
    new_stepsize = T.clip(_new_stepsize, stepsize_min, stepsize_max)
    return [(positions, new_positions),
            (stepsize, new_stepsize),
            (avg_acceptance_rate, new_acceptance_rate)]
    '''
    update = OrderedDict()
    update[positions] = new_positions.astype('float32')
    #return [(positions, new_positions)]
    return update


def wrapper_hmc(s_rng,energy_fn,dim=np.array([2,1]),L=10, beta = 0.1, epsilon = 0.1):

    """
    This should be the wrapper call that calls the various HMC definitions

    Parameters:
      Potential Energy -- function handle that captures the interest distrbution 
      Number of Leap Frog Steps -- L (10)
      Momentum corruption parameter -- beta (0.1)
      Leapfrog Integrator step length -- epsilon (0.1)
      nsamples -- The number of samples to be generated (100)
    Returns:
      samples -- Samples generated 
    """
    pos = np.random.randn(dim[0],dim[1]).astype('float32')
    vel = np.random.randn(dim[0],dim[1]).astype('float32')
    pos = theano.shared(pos,name='pos')
    vel = theano.shared(vel,name='vel')
    epsilon = theano.shared(epsilon,name='epsilon')
    L = theano.shared(L,'L')
    #pos, vel = simulate_dynamics(initial_pos=pos,initial_vel=vel,stepsize=epsilon,n_steps=L,energy_fn=energy_fn)
    accept, final_pos = hmc_move(s_rng=s_rng, positions=pos, energy_fn=energy_fn,stepsize=epsilon,n_steps=L)
    #Simulate updates
    simulate_updates = hmc_updates(positions=pos,final_pos=final_pos,accept=accept)
    simulate = theano.function([],[accept,simulate_updates[pos]],updates=simulate_updates)

    return simulate


def autocorrelation():
    X = T.tensor3().astype('float32')
    shape = X.shape
    #Assumes Length T, need to have a switch that also deals with (T/2)-1
    t_gap = T.arange(1,shape[2]-1)
    outputs_info = T.zeros((1,1,1))

    #function def that computes the mean ac for each time lag
    def calc_ac(t_gap,output_t,X):
        return T.mean(X[:,:,:-t_gap]*X[:,:,t_gap:],dtype='float32',keepdims=True)

    #We will write a scan function that loops over the indices of the data tensor
    #and computes the autocorrelation
    result,updates = theano.scan(fn= calc_ac,
    #ac,updates = theano.scan(fn= lambda X,t_gap,ac: T.mean(X[:,:,:-t_gap]*X[:,:t_gap:]),
            outputs_info=[outputs_info],
            sequences=[t_gap],
            non_sequences=[X])
    #Append zero mean value of X to the front of the array and then return 
    #Also, need to divide by the first element to scale the variances
    #For now though, let's do this in the main script
    theano_ac = theano.function(inputs=[X],outputs=[result],updates=updates)
    return theano_ac

def normed_autocorrelation(df):
    theano_ac = autocorrelation()

    Time = len(df)
    N, nbatch = df.loc[0]['X'].shape

    X = np.zeros((N,nbatch,Time))

    for tt in range(Time):
       X[:,:,tt] = df.loc[tt]['X']
    
    ac= theano_ac(X.astype('float32'))
    X_mean = np.mean(X**2,keepdims=True)[0][0]
    ac_squeeze = np.squeeze(ac[0])
    ac_squeeze = ac_squeeze/X_mean
    ac = np.vstack((1.,ac_squeeze.reshape(Time-2,1)))
    #This drops the last sample out of the data frame. Unclear, if this is the best way to do things but
    #it is the only way we can align the total number of samples from sample generation to 
    #computing autocorrelation
    ac_df = df[:-1]
    ac_df.loc[:,'autocorrelation'] = ac
    return ac_df[['num energy', 'num grad', 'autocorrelation']]
