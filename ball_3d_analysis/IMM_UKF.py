import numpy as np
from copy import deepcopy
from scipy.linalg import cholesky, inv, expm, block_diag
from get_court_position_methods import calibrate_camera, get_3d_position
from scipy.stats import multivariate_normal
import sys


class UnscentedKalmanFilter(object):
    def __init__(self, name, dim_x, dim_z, dt, fx, hx, points, sqrt_fn=None, x_mean_fn=None,
                 z_mean_fn=None, residual_x=None, residual_z=None):
        self.name = name
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
        self._log_likelihood = np.log(sys.float_info.min)
        self._mahalanobis = None

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
        self.P = (self.P + self.P.T) / 2.0
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
        elif np.isscalar(R):
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

    @property
    def log_likelihood(self):
        """
        log-likelihood of the last measurement.
        """
        if self._log_likelihood is None:
            self._log_likelihood = logpdf(x=self.y, cov=self.S)
        return self._log_likelihood


    @property
    def mahalanobis(self):
        """"
        Mahalanobis distance of measurement. E.g. 3 means measurement
        was 3 standard deviations away from the predicted value.
        """
        if self._mahalanobis is None:
            self._mahalanobis = np.sqrt(float(np.dot(np.dot(self.y.T, self.SI), self.y)))
        return self._mahalanobis

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
        self.mu = np.asarray(mu) / np.sum(mu)
        self.M = M

        x_shape = filters[0].x.shape
        for f in filters:
            if x_shape != f.x.shape:
                raise ValueError(
                    'All filters must have the same state dimension')

        self.x = np.zeros(filters[0].x.shape)
        self.P = np.zeros(filters[0].P.shape)
        self.N = len(filters)
        self.likelihood = np.zeros(self.N)
        self.omega = np.zeros((self.N, self.N))
        self._compute_mixing_probabilities()

        # initialize imm state estimate based on current filters
        self._compute_state_estimate()
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
        self._log_likelihood = np.log(sys.float_info.min)
        self._mahalanobis = None

    def filter_setx(self):
        for f in self.filters:
            f.x = self.x

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
        sum_mu = np.sum(self.mu)
        if sum_mu == 0.0:
            self.mu = self.cbar.copy()
        else:
            self.mu /= sum_mu

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
            x = np.zeros(self.x.shape)
            for kf, wj in zip(self.filters, w):
                x += kf.x * wj
            xs.append(x)

            P = np.zeros(self.P.shape)
            for kf, wj in zip(self.filters, w):
                y = kf.x - x
                P += wj * (np.outer(y, y) + kf.P)
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
            self.P += mu * (np.outer(y, y) + f.P)

    def _compute_mixing_probabilities(self):
        # Compute the mixing probability for each filter

        self.cbar = np.dot(self.mu, self.M)
        for i in range(self.N):
            for j in range(self.N):
                self.omega[i, j] = (self.M[i, j]*self.mu[i]) / self.cbar[j]

    @property
    def log_likelihood(self):
        """
        log-likelihood of the last measurement.
        """
        if self._log_likelihood is None:
            self._log_likelihood = logpdf(x=self.y, cov=self.S)
        return self._log_likelihood


    @property
    def mahalanobis(self):
        """"
        Mahalanobis distance of measurement. E.g. 3 means measurement
        was 3 standard deviations away from the predicted value.
        """
        if self._mahalanobis is None:
            self._mahalanobis = np.sqrt(float(np.dot(np.dot(self.y.T, self.SI), self.y)))
        return self._mahalanobis


class MerweScaledSigmaPoints(object):

    """
    Generates sigma points and weights according to Van der Merwe's
    2004 dissertation[1] for the UnscentedKalmanFilter class. It
    parametizes the sigma points using alpha, beta, kappa terms, and
    is the version seen in most publications.

    :param n : int
        Dimensionality of the state. 2n+1 weights will be generated.

    :param alpha : float
        Determins the spread of the sigma points around the mean.
        Usually a small positive value (1e-3).

    :param beta : float
        Incorporates prior knowledge of the distribution of the mean. For
        Gaussian x beta=2 is optimal.

    :param kappa : float, default=0.0
        Secondary scaling parameter usually set to 0 or to 3-n.

    :param sqrt_method : function(ndarray), default=scipy.linalg.cholesky
        Defines how we compute the square root of a matrix, which has
        no unique answer. Cholesky is the default choice due to its
        speed(another good choice - scipy.linalg.sqrtm).

    :param subtract : callable (x, y), optional
        Function that computes the difference between x and y.
        You will have to supply this if your state variable cannot support
        subtraction, such as angles (359-1 degreees is 2, not 358). x and y
        are state vectors, not scalars.

    Attributes:
    Wm : np.array
        weight for each sigma point for the mean
    Wc : np.array
        weight for each sigma point for the covariance
    """


    def __init__(self, n, alpha, beta, kappa, sqrt_method=None, subtract=None):
        #pylint: disable=too-many-arguments

        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        if sqrt_method is None:
            self.sqrt = cholesky
        else:
            self.sqrt = sqrt_method

        if subtract is None:
            self.subtract = np.subtract
        else:
            self.subtract = subtract

        self._compute_weights()


    def num_sigmas(self):
        """ Number of sigma points for each variable in the state x"""
        return 2*self.n + 1


    def sigma_points(self, x, P):
        """ Computes the sigma points for an unscented Kalman filter
        given the mean (x) and covariance(P) of the filter.
        Returns tuple of the sigma points and weights.

        Works with both scalar and array inputs:
        sigma_points (5, 9, 2) # mean 5, covariance 9
        sigma_points ([5, 2], 9*eye(2), 2) # means 5 and 2, covariance 9I


        :param x : An array-like object of the means of length n
        :param P : scalar, or np.array
           Covariance of the filter. If scalar, is treated as eye(n)*P.

        :return np.array, of size (n, 2n+1)
            Two dimensional array of sigma points. Each column contains all of
            the sigmas for one dimension in the problem space.

            Ordered by Xi_0, Xi_{1..n}, Xi_{n+1..2n}
        """

        if self.n != np.size(x):
            raise ValueError("expected size(x) {}, but size is {}".format(
                self.n, np.size(x)))

        n = self.n

        if np.isscalar(x):
            x = np.asarray([x])

        if  np.isscalar(P):
            P = np.eye(n)*P
        else:
            P = np.atleast_2d(P)

        lambda_ = self.alpha**2 * (n + self.kappa) - n
        U = self.sqrt((lambda_ + n)*P)

        sigmas = np.zeros((2*n+1, n))
        sigmas[0] = x
        for k in range(n):
            # pylint: disable=bad-whitespace
            sigmas[k+1]   = self.subtract(x, -U[k])
            sigmas[n+k+1] = self.subtract(x, U[k])

        return sigmas


    def _compute_weights(self):
        """ Computes the weights for the scaled unscented Kalman filter."""

        n = self.n
        lambda_ = self.alpha**2 * (n +self.kappa) - n

        c = .5 / (n + lambda_)
        self.Wc = np.full(2*n + 1, c)
        self.Wm = np.full(2*n + 1, c)
        self.Wc[0] = lambda_ / (n + lambda_) + (1 - self.alpha**2 + self.beta)
        self.Wm[0] = lambda_ / (n + lambda_)



    def __repr__(self):

        return '\n'.join([
            'MerweScaledSigmaPoints object',
            pretty_str('n', self.n),
            pretty_str('alpha', self.alpha),
            pretty_str('beta', self.beta),
            pretty_str('kappa', self.kappa),
            pretty_str('Wm', self.Wm),
            pretty_str('Wc', self.Wc),
            pretty_str('subtract', self.subtract),
            pretty_str('sqrt', self.sqrt)
        ])

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


def Q_discrete_white_noise(dim, dt=1., var=1., block_size=1, order_by_dim=True):
    """
    Returns the Q matrix for the Discrete Constant White Noise
    Model. dim may be either 2, 3, or 4 dt is the time step, and sigma
    is the variance in the noise.

    Q is computed as the G * G^T * variance, where G is the process noise per
    time step. In other words, G = [[.5dt^2][dt]]^T for the constant velocity
    model.

    :param dim : int (2, 3, or 4)
        dimension for Q, where the final dimension is (dim x dim)

    :param dt : float, default=1.0
        time step in whatever units your filter is using for time. i.e. the
        amount of time between innovations

    :param var : float, default=1.0
        variance in the noise

    :param block_size : int >= 1
        If your state variable contains more than one dimension, such as
        a 3d constant velocity model [x x' y y' z z']^T, then Q must be
        a block diagonal matrix.

    :param order_by_dim : bool, default=True
        Defines ordering of variables in the state vector. `True` orders
        by keeping all derivatives of each dimensions)
        [x x' x'' y y' y'']
        whereas `False` interleaves the dimensions
        [x y z x' y' z' x'' y'' z'']
    """

    if not (dim == 2 or dim == 3 or dim == 4):
        raise ValueError("dim must be between 2 and 4")

    if dim == 2:
        Q = [[.25*dt**4, .5*dt**3],
             [ .5*dt**3,    dt**2]]
    elif dim == 3:
        Q = [[.25*dt**4, .5*dt**3, .5*dt**2],
             [ .5*dt**3,    dt**2,       dt],
             [ .5*dt**2,       dt,        1]]
    else:
        Q = [[(dt**6)/36, (dt**5)/12, (dt**4)/6, (dt**3)/6],
             [(dt**5)/12, (dt**4)/4,  (dt**3)/2, (dt**2)/2],
             [(dt**4)/6,  (dt**3)/2,   dt**2,     dt],
             [(dt**3)/6,  (dt**2)/2 ,  dt,        1.]]

    if order_by_dim:
        return block_diag(*[Q]*block_size) * var
    return order_by_derivative(np.array(Q), dim, block_size) * var

def logpdf(x, mean=None, cov=1, allow_singular=True):
    """
    Computes the log of the probability density function of the normal
    N(mean, cov) for the data x. The normal may be univariate or multivariate.
    """

    if mean is not None:
        flat_mean = np.asarray(mean).flatten()
    else:
        flat_mean = None

    flat_x = np.asarray(x).flatten()

    if _support_singular:
        return multivariate_normal.logpdf(flat_x, flat_mean, cov, allow_singular)
    return multivariate_normal.logpdf(flat_x, flat_mean, cov)


def fx_ballistic(x, dt):
    """Балістична нелінійна модель: x_next = F*x * gravity_effect
    F-матриця для [x, vx, y, vy, z, vz]"""
    vx, vy, vz = x[1], x[3], x[5]
    vx = np.clip(vx, -40.0, 40.0)
    vy = np.clip(vy, -40.0, 40.0)
    vz = np.clip(vz, -40.0, 40.0)

    F = np.array([[1, dt, 0, 0,  0,  0],                  # x
                  [0, 1,  0, 0,  0,  0],                  # vx
                  [0, 0,  1, dt, 0,  0],                  # y
                  [0, 0,  0, 1,  0,  0],                  # vy
                  [0, 0,  0, 0,  1,  dt],                 # z
                  [0, 0,  0, 0,  0,  1]], dtype=float)    # vz

    B = np.diag([0.5 * dt**2, dt, 0.5 * dt**2, dt, 0.5 * dt**2, dt])
    k = 0.0314
    v_mag = np.sqrt(x[1]**2 + x[3]**2 + x[5]**2)
    a_dx = -k * v_mag * vx
    a_dy = -k * v_mag * vy
    a_dz = -k * v_mag * vz
    u = np.array([a_dx, a_dx, a_dy - 9.81, a_dy - 9.81, a_dz, a_dz])

    x_next = np.dot(F, x) + np.dot(B, u)

    return x_next

def fx_hit(x, dt):
    """
    Модель удару: Constant Velocity.
    Стан: [x, vx, y, vy, z, vz]
    """
    F = np.array([[1, dt, 0, 0,  0,  0],
                  [0, 1,  0, 0,  0,  0],
                  [0, 0,  1, dt, 0,  0],
                  [0, 0,  0, 1,  0,  0],
                  [0, 0,  0, 0,  1,  dt],
                  [0, 0,  0, 0,  0,  1]], dtype=float)

    return np.dot(F, x)

def fx_bounce(x, dt):
    """Модель відскоку, марковська матриця"""
    epsilon = 0.75 # Коефіцієнт реституції
    friction = 0.85 # Коефіцієнт тертя
    x_next = np.copy(x)
    vx_new = x[1] * friction
    vy_new = -x[3] * epsilon
    vz_new = x[5] * friction

    x_next[0] += vx_new * dt
    x_next[2] += vy_new * dt
    x_next[4] += vz_new * dt

    x_next[1] = vx_new
    x_next[3] = vy_new
    x_next[5] = vz_new

    return x_next

def hx(x):
    return np.array([x[0], x[2], x[4]])

def measurement_transform(prediction_data, K, R_matrix, camera_pos, ball_diameter=0.21):
    u_cam, v_cam, w_cam = prediction_data['x_pos'], prediction_data['y_pos'], prediction_data['w_box']
    raw_x, raw_y, raw_z = get_3d_position(u_cam, v_cam, w_cam, K, R_matrix, camera_pos, ball_diameter)
    raw_y = -raw_y
    measurement = np.array([raw_x, raw_y, raw_z], dtype=np.float32)

    return measurement

def get_dynamic_transition_matrix(y, vy):
    """
    Повертає марковську матрицю 3x3 для моделей: [Ballistic, Hit, Bounce]
    """
    M = np.array([[0.95, 0.04, 0.01],
                  [0.60, 0.40, 0.00],
                  [0.90, 0.00, 0.10]])

    if y < 0.3 and vy < 0:
        M[0] = [0.10, 0.05, 0.85]
        M[1] = [0.10, 0.05, 0.85]

    return M