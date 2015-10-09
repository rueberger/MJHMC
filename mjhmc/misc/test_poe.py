from mjhmc.misc import distributions
import numpy as np

dims = 36
batch = 100

X = np.random.randn(dims,batch)

poe2 = distributions.ProductOfT2(nbatch=1)
poe = distributions.ProductOfT(nbatch=1, W=poe2.W, logalpha=poe2.logalpha)

for ii in xrange(5):
    print ii
    X = np.random.randn(dims, batch)
    print poe.E_val(X)
    print poe2.E_val(X)
    print poe.dEdX_val(X)
    print poe2.dEdX_val(X)
    assert poe.E_val(X) == poe2.E_val(X)
    assert poe.dEdX_val(X) == poe2.dEdX_val(X)
