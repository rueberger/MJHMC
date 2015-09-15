# Markov Jump Hamiltonian Monte Carlo
Python implementation of Markov Jump HMC

Markov Jump HMC is described in the paper

> Berger, Andrew and Mudigonda, Mayur and DeWeese, Michael R. and Sohl-Dickstein, Jascha<br>
> A Markov Jump Process for More Efficient Hamiltonian Monte Carlo <br>
> http://arxiv.org/abs/1509.03808

## Example Python Code

```python
from mjhmc.samplers.markov_jump_hmc import MarkovJumpHMC
import numpy as np

# Define the energy function and gradient
def E(X, sigma=1.):
    """ Energy function for isotropic Gaussian """
    return np.sum(X**2, axis=0).reshape((1,-1))/2./sigma**2
    
def dEdX(X, sigma=1.):
    """ Energy function gradient for isotropic Gaussian """
    return X/sigma**2

# Initialize the sample locations -- 2 dimensions, 100 particles
Xinit = np.random.randn(2,100)

# initialize the sampler.
sampler = MarkovJumpHMC(Xinit, E, dEdX, epsilon=0.1, beta=0.1)
# perform 10 sampling steps for all 100 particles
X = sampler.sample(num_steps = 10)
# perform another 10 sampling steps
X = sampler.sample(num_steps = 10)
```
