from mjhmc.misc import distributions
import numpy as np

dims = 36
batch = 100

X = np.random.randn(dims,batch)

poe = distributions.ProductOfT()
print(poe.E_val(X))
print(poe.dEdX_val(X))
print(poe.dEdX_val(X).shape)
