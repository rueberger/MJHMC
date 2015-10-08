import theano
import theano.tensor as T
import numpy as np
 
 
### create a function using theano that computes the product
### of experts energy function and gradient
def E_def(X, W, b, logalpha):
        """
        energy for a POE with student's-t expert in terms of:
                samples [# samples]x[# dimensions] X
                receptive fields [# dimensions]x[# experts] W
                biases [# experts] b
                expert weighting [# experts] alpha
        """
        b = b.reshape((1,-1))
        alpha = T.exp(logalpha).reshape((1,-1))
        E_perexpert = alpha*T.log(1 + (T.dot(X, W) + b)**2)
        E = T.sum(E_perexpert)
        return E
 
## initialize parameters
b = T.vector()
logalpha = T.vector()
W = T.matrix()
X = T.matrix()
 
E = E_def(X, W, b, logalpha)
dEdX = T.grad(E, X)
 
E_func = theano.function([X, W, b, logalpha], E, allow_input_downcast=True)
dEdX_func = theano.function([X, W, b, logalpha], dEdX, allow_input_downcast=True)
 
 
### set model parameters
n_d = 36 # data dimensions
n_e = n_d*2 # experts
n_batch = 100 # samples
 
W = np.random.randn(n_d, n_e) # weight matrix
b = np.random.randn(n_e,) # bias
logalpha = np.random.randn(n_e,) # log alpha
 
### initialize samples
X_init = np.random.randn(n_batch, n_d)
 
## test computation of energy function and gradient
print "Energy", E_func(X_init, W, b, logalpha)
def stats(u):
  return dict(min=np.min(u), mean=np.mean(u), med=np.median(u), max=np.max(u), std=np.std(u), notfin=np.sum(~np.isfinite(u)), shp=np.asarray(u).shape)
print "dEdX stats", stats(dEdX_func(X_init, W, b, logalpha))
