import numpy as np
from .utils import overrides
import theano.tensor as T
import theano
from scipy.sparse import rand

class Distribution(object):
    """
    interface for distributions
    """

    def __init__(self, ndims=2, nbatch=100):
        self.ndims = ndims
        self.nbatch = nbatch
        self.init_X()
        self.E_count = 0
        self.dEdX_count = 0

    def E(self, X):
        self.E_count += X.shape[1]
        return self.E_val(X)

    def E_val(self, X):
        raise NotImplementedError()

    def dEdX(self, X):
        self.dEdX_count += X.shape[1]
        return self.dEdX_val(X)

    def dEdX_val(self, X):
        raise NotImplementedError()

    def init_X(self):
        """
        Sets self.Xinit to a good initial value
        """
        raise NotImplementedError()

    def reset(self):
        """
        resets the object. returns self for convenience
        """
        self.E_count = 0
        self.dEdX_count = 0
        self.init_X()
        return self

    def __call__(self, X):
        """
        Convenience method for NUTS compatibility
        returns -E, -dEdX
        """
        rshp_X = X.reshape(len(X), 1)
        E = float(self.E(rshp_X))
        dEdX = self.dEdX(rshp_X).T[0]
        return -E, -dEdX

class Gaussian(Distribution):
    def __init__(self, ndims=2, nbatch=100, log_conditioning=6):
        """
        Energy function, gradient, and hyperparameters for the "ill
        conditioned Gaussian" example from the LAHMC paper.
        """
        self.conditioning = 10**np.linspace(-log_conditioning, 0, ndims)
        self.J = np.diag(self.conditioning)
        self.description = '%dD Anisotropic Gaussian, %g self.conditioning'%(ndims, 10**log_conditioning)
        super(Gaussian, self).__init__(ndims, nbatch)

    @overrides(Distribution)
    def E_val(self, X):
        return np.sum(X*np.dot(self.J,X), axis=0).reshape((1,-1))/2.

    @overrides(Distribution)
    def dEdX_val(self, X):
        return np.dot(self.J,X)/2. + np.dot(self.J.T,X)/2.

    @overrides(Distribution)
    def init_X(self):
        self.Xinit = (1./np.sqrt(self.conditioning).reshape((-1,1))) * np.random.randn(self.ndims,self.nbatch)

class RoughWell(Distribution):
    def __init__(self, ndims=2, nbatch=100, scale1=100, scale2=4):
        """
        Energy function, gradient, and hyperparameters for the "rough well"
        example from the LAHMC paper.
        """
        self.scale1 = scale1
        self.scale2 = scale2
        self.description = '{} Rough Well'.format(ndims)
        super(RoughWell, self).__init__(ndims, nbatch)

    @overrides(Distribution)
    def E_val(self, X):
        cosX = np.cos(X*2*np.pi/self.scale2)
        E = np.sum((X**2) / (2*self.scale1**2) + cosX, axis=0).reshape((1,-1))
        return E

    @overrides(Distribution)
    def dEdX_val(self, X):
        sinX = np.sin(X*2*np.pi/self.scale2)
        dEdX = X/self.scale1**2 + -sinX*2*np.pi/self.scale2
        return dEdX

    @overrides(Distribution)
    def init_X(self):
        self.Xinit = self.scale1 * np.random.randn(self.ndims, self.nbatch)

class MultimodalGaussian(Distribution):
    def __init__(self, ndims=2, nbatch=100, separation=3):
        self.sep_vec = np.array([separation] * nbatch +
                                [0] * (ndims - 1) * nbatch).reshape(ndims, nbatch)
        # separated along first axis
        self.sep_vec[0] += separation
        super(MultimodalGaussian, self).__init__(ndims, nbatch)

    @overrides(Distribution)
    def E_val(self, X):
        trim_sep_vec = self.sep_vec[:, :X.shape[1]]
        return -np.log(np.exp(-np.sum((X + trim_sep_vec)**2, axis=0)) +
                       np.exp(-np.sum((X - trim_sep_vec)**2, axis=0)))

    @overrides(Distribution)
    def dEdX_val(self, X):
        # allows for partial batch size
        trim_sep_vec = self.sep_vec[:, :X.shape[1]]
        common_exp = np.exp(np.sum(4 * trim_sep_vec * X, axis=0))
        # floating point hax
        return ((2 * ((X - trim_sep_vec) * common_exp + trim_sep_vec + X)) /
                (common_exp + 1))


    @overrides(Distribution)
    def init_X(self):
        # okay, this is pointless... sep vecs cancel
        self.Xinit = ((np.random.randn(self.ndims, self.nbatch) + self.sep_vec) +
                (np.random.randn(self.ndims, self.nbatch) - self.sep_vec))

class TestGaussian(Distribution):

    def __init__(self, ndims=2, nbatch=100, sigma=1.):
        """Simple default unit variance gaussian for testing samplers
        """
        super(TestGaussian, self).__init__(ndims, nbatch)
        self.sigma = sigma

    @overrides(Distribution)
    def E_val(self, X):
        return np.sum(X**2, axis=0).reshape((1,-1))/2./self.sigma**2

    @overrides(Distribution)
    def dEdX_val(self, X):
        return X/self.sigma**2

    @overrides(Distribution)
    def init_X(self):
        self.Xinit = np.random.randn(self.ndims, self.nbatch)

class ProductOfT(Distribution):

    def __init__(self,ndims=36,nbasis=72,nbatch=100,logalpha=None,W=None,b=None):
        """ Product of T experts, assumes a fixed W that is sparse and alpha that is
        """
        self.ndims=ndims
        self.nbasis=nbasis
        self.nbatch=nbatch
        if W is  None:
           rand_val = rand(ndims,nbasis/2,density=0.25)
           W = np.concatenate([rand_val.toarray(), -rand_val.toarray()],axis=1)
        self.W = theano.shared(np.array(W,dtype='float32'),'W')
        if logalpha is None:
            logalpha = np.random.randn(nbasis,)
        self.logalpha = theano.shared(np.array(logalpha,dtype='float32'),'alpha')
        if b is None:
            b = np.zeros((nbasis,))
        self.b = theano.shared(np.array(b,dtype='float32'),'b')
        X = T.matrix()
        E = self.E_def(X)
        dEdX = T.grad(T.sum(E),X)
        #@overrides(Distribution)
        self.E_val=theano.function([X],E,allow_input_downcast=True)
        #@overrides(Distribution)
        self.dEdX_val = theano.function([X],dEdX,allow_input_downcast=True)
        super(ProductOfT,self).__init__(ndims,nbatch)

    def E_def(self,X):
        """
        energy for a POE with student's-t expert in terms of:
                samples [# samples]x[# dimensions] X
                receptive fields [# dimensions]x[# experts] W
                biases [# experts] b
                expert weighting [# experts] alpha
        """
        self.b = self.b.reshape((1,-1))
        alpha = T.exp(self.logalpha).reshape((1,-1))
        E_perexpert = alpha*T.log(1 + (T.dot(X.T,self.W) + self.b)**2)
        E = T.sum(E_perexpert, axis=1).reshape((1,-1))
        return E

    """
    @overrides(Distribution)
    def dEdX_val(self,X):
        dEdX = np.sum(((2*self.alpha*(self.W.T,X))/(1+(self.W.T,X)**2))*self.W,axis=0)
        return dEdX
    """

    @overrides(Distribution)
    def init_X(self):
        self.Xinit = np.random.randn(self.ndims,self.nbatch)


class ProductOfT2(Distribution):

    def __init__(self,ndims=36,nbasis=72,nbatch=100,logalpha=None,W=None):
        """ Product of T experts, assumes a fixed W that is sparse and alpha that is*
        """
        if W is  None:
           rand_val = rand(ndims,nbasis/2,density=0.25)
           self.W = np.concatenate([rand_val.toarray(), -rand_val.toarray()],axis=1)
        else:
           self.W = W

        self.logalpha = logalpha or np.random.randn(nbasis, 1)
        self.alpha = np.exp(self.logalpha)
        super(ProductOfT2,self).__init__(ndims,nbatch)
        '''
        E = \sum_i f( \sum_j W_ij x_j )
        f(u) = alpha log( 1 + u^2 )
        alpha = 1, or 2, or something*
        '''

    def f(self,u):
        return self.alpha*np.log(1 + u**2)

    @overrides(Distribution)
    def E_val(self,X):
        E = np.sum(self.f(np.dot(self.W.T,X)))
        return E

    @overrides(Distribution)
    def dEdX_val(self,X):
        num = (2*self.alpha * np.dot(self.W.T,X))
        denom = (1 + np.dot(self.W.T,X)**2)
        dEdX = np.sum(np.dot((num / denom), self.W.T), axis=0)
        return dEdX

    @overrides(Distribution)
    def init_X(self):
        self.Xinit = np.random.randn(self.ndims,self.nbatch)
