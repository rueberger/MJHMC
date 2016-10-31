"""
This module contains the TensorflowDistribution base class and distributions that inherit from it
"""

import numpy as np
import tensorflow as tf
from .utils import overrides, package_path
import os
from scipy import stats
from scipy.io import loadmat
import pickle
from math import sqrt


from mjhmc.misc.distributions import Distribution


class TensorflowDistribution(Distribution):
    """ Base class for distributions defined by energy functions written
    in Tensorflow

    You should give your TensorflowDistribution objects a name. Use a
    descriptive name, and use the same for functionally equivalent
    TensorflowDistributions - the hash of the name is used to label the
    initialization information which is generated at first run time of
    a new distribution. This requirement is a side effect of the
    unfortunate fact that there is no computable hash function which
    assigns functionally identical programs to the same number.

    TensorflowDistribution is subclassed by defining the distribution energy op in
    build_energy_op
    """

    #pylint: disable=too-many-arguments
    def __init__(self, name=None, sess=None, device='/cpu:0'):
        """ Creates a TensorflowDistribution object

        ndims and nbatch are inferred from init
        nbatch must match shape of energy_op

        :param name: name of this distribution. use the same name for functionally identical distributions
        :param sess: optional session. If none, one will be created
        :param device: device to execute tf ops on. By default uses cpu to avoid compatibility issues
        :returns: TensorflowDistribution object
        :rtype: TensorflowDistribution
        """
        self.graph = tf.Graph()
        self.device = device
        with self.graph.as_default(), tf.device(self.device):
            self.sess = sess or tf.Session()
            self.build_graph()

        self.name = name or self.energy_op.op.name

        super(TensorflowDistribution, self).__init__(ndims=self.ndims, nbatch=self.nbatch)

        self.backend = 'tensorflow'
        super(TensorflowDistribution, self).__init__(ndims=self.ndims, nbatch=self.nbatch)


    def build_graph(self):
        with self.graph.as_default(), tf.device(self.device):
            self.state_pl = tf.placeholder(tf.float32, [self.ndims, None])
            self.build_energy_op()

            self.grad_op = tf.gradients(self.energy_op, self.state_pl)[0]
            self.sess.run(tf.initialize_all_variables())


    def build_energy_op(self):
        """ Sets self.energy_op

        """
        raise NotImplementedError("this method must be defined to subclass TensorflowDistribution")

    def grad_op_singlet(self, singlet_state):
        """ Build the gradient op for a single particle

        Args:
          singlet_state: tensor with state of single particle - [n_dims]
        """
        return tf.gradients(self.energy_op_singlet(singlet_state), singlet_state)[0]


    def build_energy_op(self):
        """ Builds an energy op over all particles use self.singlet_energy_op and tf.map_fn
        Also builds the gradient op
        """
        # TODO: unclear what's the right number of parallel_itrs
        # also unclear if swapping memory helps
        # unpacks along first dimension
        # [self.state_pl.shape[1]]
        self.energy_op = tf.map_fn(self.energy_op_singlet, tf.transpose(self.state_pl),
                           back_prop=False,
                           swap_memory=False,
                           name='energy')
        # [self.state_pl.shape[1], ndims]
        self.grad_op = tf.map_fn(self.grad_op_singlet, tf.transpose(self.state_pl),
                                 back_prop=False,
                                 swap_memory=False,
                                 name='grad_T')
        self.grad_op = tf.transpose(self.grad_op, name='grad')


    @overrides(Distribution)
    def E_val(self, X):
        with self.graph.as_default(), tf.device(self.device):
            energy = self.sess.run(self.energy_op, feed_dict={self.state_pl: X})
            return energy

    @overrides(Distribution)
    def dEdX_val(self, X):
        with self.graph.as_default(), tf.device(self.device):
            grad = self.sess.run(self.grad_op, feed_dict={self.state_pl: X})
            return grad

    @overrides(Distribution)
    def __hash__(self):
        return hash((self.ndims, self.name))

class Funnel(TensorflowDistribution):
    """ This class implements the Funnel distribution as specified in Neal, 2003
    In particular:
      x_0 ~ N(0, scale^2)
      x_i ~ N(0, e^x_0); i in {1, ... ,ndims}
    """

    def __init__(self,scale=1.0, nbatch=50, ndims=10, **kwargs):
        self.scale = float(scale)
        self.ndims = ndims
        self.nbatch = nbatch
        self.gen_init_X()

        super(Funnel, self).__init__(name='Funnel', **kwargs)

    @overrides(TensorflowDistribution)
    def energy_op_singlet(self, singlet_state):
        with self.graph.as_default(), tf.device(self.device):
            # [1, nbatch]
            e_x_0 = tf.neg((self.state_pl[0, :] ** 2) / (self.scale ** 2), name='E_x_0')
            # [ndims - 1, nbatch]
            e_x_k = tf.neg((self.state_pl[1:, :] ** 2) / tf.exp(self.state_pl[0, :]), name='E_x_k')
            # [nbatch]
            self.energy_op = tf.reduce_sum(tf.add(e_x_0, e_x_k), 0, name='energy_op')



    @overrides(Distribution)
    def gen_init_X(self):
        x_0 = np.random.normal(scale=self.scale, size=(1, self.nbatch))
        x_k = np.random.normal(scale=np.exp(x_0), size=(self.ndims - 1, self.nbatch))
        self.Xinit = np.vstack((x_0, x_k))

    @overrides(Distribution)
    def __hash__(self):
        return hash((self.scale, self.ndims))

class TFGaussian(TensorflowDistribution):
    """ Standard gaussian implemented in tensorflow
    """
    def __init__(self, ndims=2, nbatch=100, sigma=1.,  **kwargs):
        self.ndims  = ndims
        self.nbatch = nbatch
        self.sigma = sigma
        self.gen_init_X()

        super(TFGaussian, self).__init__(name='TFGaussian', **kwargs)

    @overrides(TensorflowDistribution)
    def energy_op_singlet(self, singlet_state):
        with self.graph.as_default(), tf.device(self.device):
            self.energy_op = tf.reduce_sum(self.state_pl ** 2, 0) / (2 * self.sigma ** 2)


    @overrides(Distribution)
    def gen_init_X(self):
        self.Xinit = np.random.randn(self.ndims, self.nbatch)

    @overrides(Distribution)
    def __hash__(self):
        return hash((self.ndims, self.sigma))

class SparseImageCode(TensorflowDistribution):
    """ Distribution over the coefficients in an inference model of sparse coding on natural images a la Olshausen and Field
    """

    def __init__(self, n_patches, n_batches=10, **kwargs):
        """ Construct a SparseImageCode object

        Args:
           n_patches: number of patches to simultaneously run inference over - must be a perfect square
           n_batches: number of batches to run at once
        """
        patch_size = 16
        self.lmbda = 0.01

        assert int(sqrt(n_patches)) == sqrt(n_patches)

        img_path = os.path.expanduser('~/data/mjhmc/IMAGES.mat')
        basis_path = os.path.expanduser('~/data/mjhmc/basis.mat')

        # [512, 512, 10]
        self.imgs = loadmat(img_path)['IMAGES']
        # [256, 256], [img_size, n_coeffs]
        self.basis = loadmat(basis_path)['basis']

        # n_coeffs per patch
        self.img_size, self.n_coeffs = self.basis.shape
        self.ndims = n_patches * self.n_coeffs
        self.nbatch = n_batches
        self.n_patches = n_patches
        sqrt_n_patches = int(sqrt(n_patches))

        # select a square set of patches from the image starting from the upper left
        self.patch_list = []
        for x_idx in range(int(sqrt(n_patches))):
            for y_idx in range(int(sqrt(n_patches))):
                self.patch_list.append(self.imgs[x_idx * patch_size: (x_idx + 1) * patch_size, y_idx * patch_size: (y_idx + 1) * patch_size, 0].ravel())

        super(SparseImageCode, self).__init__(name='SparseImageCode', **kwargs)


    @overrides(TensorflowDistribution)
    def build_energy_op(self):
        with self.graph.as_default(), tf.device(self.device):
            n_active = tf.shape(self.state_pl)[1]
            # [n_patches, 1, img_size]
            self.patches = tf.to_float(tf.reshape(tf.pack(self.patch_list),
                                                  [self.n_patches, 1, self.img_size]),                                       name='patches')
            shaped_state = tf.reshape(self.state_pl, [self.n_patches, n_active, self.n_coeffs, 1], name='shaped_state')
            shaped_basis = tf.reshape(self.basis, [1, 1, self.img_size, self.n_coeffs], name='shaped_basis')
            # [n_patches, nbatch, img_size, n_coeffs]
            shaped_basis = tf.tile(shaped_basis, [self.n_patches, n_active, 1, 1], name='tiled_basis')
            # [n_patches, nbatch, img_size]
            reconstructions = tf.squeeze(tf.batch_matmul(shaped_basis, shaped_state), name='reconstructions')
            # [n_patches, nbatch]
            reconstruction_error = tf.reduce_sum(0.5 * (self.patches - reconstructions) ** 2, -1)
            # [nbatch]
            reconstruction_error = tf.reduce_mean(reconstruction_error, 0, name='reconstruction_error')

            # [nbatch]
            sp_penalty = self.lmbda * tf.reduce_sum(tf.abs(self.state_pl), 0, name='sp_penalty')
            self.energy_op = reconstruction_error + sp_penalty

    @overrides(Distribution)
    def __hash__(self):
        # so they can be hashed
        self.imgs.flags.writeable = False
        self.basis.flags.writeable = False
        return hash((hash(self.imgs.data),
                     hash(self.basis.data),
                     hash(self.lmbda),
                     hash(self.n_patches)))
