from mjhmc.figures import ac_fig
from mjhmc.misc.distributions import ProductOfT
import numpy as np

np.random.seed(2015)


#Search for best hyper-parameters

#Parameters for the distribution object
ndims = 36
nbasis = 36
nbatch = 25
POT = ProductOfT(nbasis=nbasis,nbatch=nbatch,ndims=ndims)

#Run a comparison
ac_fig.plot_best(POT,num_steps=100000,update_params=True)

