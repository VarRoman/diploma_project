import numpy as np
from copy import deepcopy
from scipy.linalg import cholesky
from scipy.stats import multivariate_normal


class UnscentedKalmanFilter(object):
    def __init__(self, dim_x, dim_z, dt, fx, hx, points, sqrt_fn=None, x_mean_fn=None, z_mean_fn=None, residual_x=None, residual_z=None):
        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x)
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)
        self.Q = np.eye(dim_x)
        self.R = np.eye(dim_z)
        self._dim_x = dim_x
        self._dim_z = dim_z
        self.points_fn = points
        self._dt = dt
        self._num_sigmas = points.num_sigmas()
        self.hx = hx
        self.fx = fx
        self.x_mean_fn = x_mean_fn
        self.z_mean_fn = z_mean_fn
        self.Wm, self.Wc = points.Wm, points.Wc
        self.likelihood = None

        if sqrt_fn is None:
            self.msqrt = cholesky
        else:
            self.msqrt = sqrt_fn

        if residual_x is None:
            self.residual_x = np.subtract
        else:
            self.residual_x = residual_x

        if residual_z is None:
            self.residual_z = np.subtract
        else:
            self.residual_z = residual_z

        # sigma points transformed through f(x) and h(x)
        self.sigmas_f = np.zeros((self._num_sigmas, self._dim_x))
        self.sigmas_h = np.zeros((self._num_sigmas, self._dim_z))

        self.K = np.zeros((dim_x, dim_z))
        self.y = np.zeros(dim_z)
        self.z = np.array([[None]*dim_z])
        self.S = np.zeros((dim_z, dim_z))
        self.SI = np.zeros((dim_z, dim_z))
        self.inv = np.linalg.inv

        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def predict(self, dt=None, UT=None, fx=None, **fxargs):
        """
        :param dt: time step for next iteration of computation
        :param UT: unscented transform function
        :param fx: state transition function
        :param fxargs: arguments passed to f(x)
        """
        if dt is None:
            dt = self._dt

        if UT is None:
            UT = unscented_transform

        # calculating sigma points for given mean and covariance
        self.compute_process_sigmas(dt, fx, **fxargs)
        #pass sigmas through the unscented transform to compute the prior
        self.x, self.P = UT(self.sigmas_f, self.Wm, self.Wc, self.Q, self.x_mean_fn, self.residual_x)
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)

    def update(self, z, R=None, UT=None, hx=None, **hxargs):
        """
        :param z: measurement vector
        :param R: measurement noise matrix
        :param UT: function for unscented transform
        :param hx: measurement function
        :param hxargs: arguments for hx
        """
        if z is None:
            self.z = np.array([[None]*self._dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            self.likelihood = 1.0
            return

        if hx is None:
            hx = self.hx

        if UT is None:
            UT = unscented_transform

        if R is None:
            R = self.R
        elif isscalar(R):
            R = np.eye(self._dim_z) * R

        # passing prior sigmas through h(x) to get measurement sigmas
        # the shape of simgas_h will vary if the shape of z varies
        sigmas_h = []
        for s in self.sigmas_f:
            sigmas_h.append(hx(s, **hxargs))

        self.sigmas_h = np.atleast_2d(sigmas_h)

        # mean and covariance of prediction passed through unscented transform
        z_mean, self.S = UT(self.sigmas_h, self.Wm, self.Wc, R, self.z_mean_fn, self.residual_z)
        self.SI = self.inv(self.S)

        # compute cross variance of the state and measurement
        Pxz = self.cross_variance(self.x, z_mean, self.sigmas_f, self.sigmas_h)

        self.K = np.dot(Pxz, self.SI) # Kalman gain
        self.y = self.residual_z(z, z_mean)
        S_stable = self.S + np.eye(self.S.shape[0]) * 1e-6
        try:
            self.likelihood = multivariate_normal.pdf(self.y, mean=np.zeros_like(self.y), cov=S_stable)
        except np.linalg.LinAlgError:
            # Catching singularity effect
            self.likelihood = 1e-300

        # updating Gaussian state estimate(x, P)
        self.x = self.x + np.dot(self.K, self.y)
        self.P = self.P - np.dot(self.K, np.dot(self.S, self.K.T))

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def cross_variance(self, x, z, sigmas_f, sigmas_h):
        # Computing cross variance of the state 'x' and measurement 'z'
        Pxz = np.zeros((sigmas_f.shape[1], sigmas_h.shape[1]))
        N = sigmas_f.shape[0]
        for i in range(N):
            dx = self.residual_x(sigmas_f[i], x)
            dz = self.residual_z(sigmas_h[i], z)
            Pxz += self.Wc[i] * np.outer(dx, dz)
        return Pxz

    def compute_process_sigmas(self, dt, fx=None, **fx_args):
        # computes the values of sigmas_f
        if fx is None:
            fx = self.fx

        # calculate sigma points for given mean and covariance
        sigmas = self.points_fn.sigma_points(self.x, self.P)

        for i, s in enumerate(sigmas):
            self.sigmas_f[i] = fx(s, dt, **fx_args)

def unscented_transform(sigmas, Wm, Wc, noise_cov=None,
                        mean_fn=None, residual_fn=None):
    r"""
    Computes unscented transform of a set of sigma points and weights.
    returns the mean and covariance in a tuple.
    :param residual_fn:
    :param sigmas: ndarray, of size (n, 2n+1)
        2D array of sigma points.

    :param Wm : ndarray [# sigmas per dimension]
        Weights for the mean.


    :param Wc : ndarray [# sigmas per dimension]
        Weights for the covariance.

    :param noise_cov: ndarray, optional
        noise matrix added to the final computed covariance matrix.

    :param mean_fn: callable (sigma_points, weights), optional
        Function that computes the mean of the provided sigma points
        and weights
    """

    kmax, n = sigmas.shape

    try:
        if mean_fn is None:
            # new mean is just the sum of the sigmas * weight
            x = np.dot(Wm, sigmas)    # dot = \Sigma^n_1 (W[k]*Xi[k])
        else:
            x = mean_fn(sigmas, Wm)
    except:
        print(sigmas)
        raise


    # new covariance is the sum of the outer product of the residuals
    # times the weights

    # this is the fast way to do this - see 'else' for the slow way
    if residual_fn is np.subtract or residual_fn is None:
        y = sigmas - x[np.newaxis, :]
        P = np.dot(y.T, np.dot(np.diag(Wc), y))
    else:
        P = np.zeros((n, n))
        for k in range(kmax):
            y = residual_fn(sigmas[k], x)
            P += Wc[k] * np.outer(y, y)

    if noise_cov is not None:
        P += noise_cov

    return x, P


class IMMEstimator(object):
    """ Implements an Interacting Multiple-Model (IMM) estimator.
    :param filters: N-list consisting of filters in sequenced order
    :param mu: (N) array-like of float with mode probabilities for eache filter
    :param M: (N, N) Markov chain transition matrix
    """

    def __init__(self, filters, mu, M):
        if len(filters) < 2:
            raise ValueError('filters must contain at least two filters')

        self.filters = filters
        self.mu = asarray(mu) / np.sum(mu)
        self.M = M

        x_shape = filters[0].x.shape
        for f in filters:
            if x_shape != f.x.shape:
                raise ValueError(
                    'All filters must have the same state dimension')

        self.x = zeros(filters[0].x.shape)
        self.P = zeros(filters[0].P.shape)
        self.N = len(filters)  # number of filters
        self.likelihood = zeros(self.N)
        self.omega = zeros((self.N, self.N))
        self._compute_mixing_probabilities()

        # initialize imm state estimate based on current filters
        self._compute_state_estimate()
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def update(self, z):
        """
        Add a new measurement (z) to the Kalman filter. If z is None, nothing
        is changed.
        :param z: measurement
        """

        # run update on each filter, and save the likelihood
        for i, f in enumerate(self.filters):
            f.update(z)
            self.likelihood[i] = f.likelihood

        # update mode probabilities from total probability * likelihood
        self.mu = self.cbar * self.likelihood
        self.mu /= np.sum(self.mu)  # normalize

        self._compute_mixing_probabilities()

        # compute mixed IMM state and covariance and save posterior estimate
        self._compute_state_estimate()
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def predict(self, u=None):
        """
        Predict next state (prior) using the IMM state propagation
        equations.
        :param u: control vector, if None then none
        """

        # compute mixed initial conditions
        xs, Ps = [], []
        for i, (f, w) in enumerate(zip(self.filters, self.omega.T)):
            x = zeros(self.x.shape)
            for kf, wj in zip(self.filters, w):
                x += kf.x * wj
            xs.append(x)

            P = zeros(self.P.shape)
            for kf, wj in zip(self.filters, w):
                y = kf.x - x
                P += wj * (outer(y, y) + kf.P)
            Ps.append(P)

        #  compute each filter's prior using the mixed initial conditions
        for i, f in enumerate(self.filters):
            # propagate using the mixed state estimate and covariance
            f.x = xs[i].copy()
            f.P = Ps[i].copy()
            f.predict(u)

        # compute mixed IMM state and covariance and save posterior estimate
        self._compute_state_estimate()
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

    def _compute_state_estimate(self):
        """
        Computes the IMM's mixed state estimate from each filter using
        the the mode probability self.mu to weight the estimates.
        """
        self.x.fill(0)
        for f, mu in zip(self.filters, self.mu):
            self.x += f.x * mu

        self.P.fill(0)
        for f, mu in zip(self.filters, self.mu):
            y = f.x - self.x
            self.P += mu * (outer(y, y) + f.P)

    def _compute_mixing_probabilities(self):
        # Compute the mixing probability for each filter

        self.cbar = dot(self.mu, self.M)
        for i in range(self.N):
            for j in range(self.N):
                self.omega[i, j] = (self.M[i, j]*self.mu[i]) / self.cbar[j]