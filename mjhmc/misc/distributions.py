import numpy as np
from .utils import overrides

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