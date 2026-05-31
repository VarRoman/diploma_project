import numpy as np
from copy import deepcopy
from scipy.linalg import cholesky, inv, expm, block_diag
from get_court_position_methods import calibrate_camera, get_3d_position
from scipy.stats import multivariate_normal
import sys


class UnscentedKalmanFilter(object):
    def __init__(self, name, dim_x, dim_z, dt, fx, hx, points, sqrt_fn=None,
            x_mean_fn=None, z_mean_fn=None, residual_x=None, residual_z=None):
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
        self.x, self.P = UT(self.sigmas_f, self.Wm, self.Wc,
                        self.Q, self.x_mean_fn, self.residual_x)
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
        z_mean, self.S = UT(self.sigmas_h, self.Wm, self.Wc,
                            R, self.z_mean_fn, self.residual_z)
        # Регуляризуємо S перед інверсією/правдоподібністю: гарантована
        # позитивна визначеність навіть при тимчасово виродженій коваріації.
        S_stable = self.S + np.eye(self.S.shape[0]) * 1e-6
        try:
            self.SI = np.linalg.inv(S_stable)
        except np.linalg.LinAlgError:
            self.SI = np.linalg.pinv(S_stable)

        # compute cross variance of the state and measurement
        Pxz = self.cross_variance(self.x, z_mean, self.sigmas_f, self.sigmas_h)

        self.K = np.dot(Pxz, self.SI) # Kalman gain
        self.y = self.residual_z(z, z_mean)
        # multivariate_normal.pdf без allow_singular=True може кидати ValueError
        # ("the input matrix must be positive semidefinite") — а не лише
        # LinAlgError, тому ловимо обидва.
        try:
            self.likelihood = multivariate_normal.pdf(
                self.y,
                mean=np.zeros_like(self.y),
                cov=S_stable,
                allow_singular=True,
            )
        except (np.linalg.LinAlgError, ValueError):
            self.likelihood = 1e-300
        # IMM-рівень очікує строго додатну правдоподібність:
        if not np.isfinite(self.likelihood) or self.likelihood <= 0.0:
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
        return (f'MerweScaledSigmaPoints(n={self.n}, alpha={self.alpha}, '
                f'beta={self.beta}, kappa={self.kappa})')

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

    return multivariate_normal.logpdf(flat_x, flat_mean,
                                      cov, allow_singular=allow_singular)


def fx_ballistic(x, dt):
    """
    9D балістична модель (CV-CA-IMM): state = [x, vx, ax, y, vy, ay, z, vz, az].

    Гравітація — driving term (-g по Y), не входить у state.
    Квадратичний drag — також driving term: a_drag = -k·|v|·v.
    State-компоненти ax, ay, az — "залишкове" прискорення: ≈0 у чистій
    балістиці, ≠0 під час удару/блоку (накопичується через великий Q[a]
    у hit-моделі). Інтерпретованість: |a_state| ↔ сила маневру.

    Дискретизація constant-acceleration:
        p_new = p + v·dt + 0.5·a_total·dt^2
        v_new = v + a_total·dt
        a_new = a (детермінованно — змінюється лише через Q[a] шум).

    Step 2.B (Phase 3): переведено з 6D CV+driving-gravity на 9D CA.
    """
    vx = np.clip(x[1], -40.0, 40.0)
    vy = np.clip(x[4], -40.0, 40.0)
    vz = np.clip(x[7], -40.0, 40.0)
    ax_s, ay_s, az_s = x[2], x[5], x[8]

    k = 0.0314
    v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
    a_drag_x = -k * v_mag * vx
    a_drag_y = -k * v_mag * vy
    a_drag_z = -k * v_mag * vz

    g = 9.81
    a_tot_x = ax_s + a_drag_x
    a_tot_y = ay_s + a_drag_y - g
    a_tot_z = az_s + a_drag_z

    return np.array([
        x[0] + vx*dt + 0.5*a_tot_x*dt**2,
        vx + a_tot_x*dt,
        ax_s,
        x[3] + vy*dt + 0.5*a_tot_y*dt**2,
        vy + a_tot_y*dt,
        ay_s,
        x[6] + vz*dt + 0.5*a_tot_z*dt**2,
        vz + a_tot_z*dt,
        az_s,
    ])


def fx_hit(x, dt):
    """
    9D модель удару: структурно ідентична fx_ballistic (CA з гравітацією
    + drag). Відмінність — у Q-матриці (create_imm_estimator): hit-mode
    має значно більшу Q[a], що дозволяє accel "прокинутись" від ~0 до
    імпульсних значень за один-два кадри. Так удар/блок "запам'ятовується"
    як ненульовий accel у state, замість випадкового дрейфу швидкості
    через великий Q[v] у старій 6D-версії.

    Це класична CV-CA-IMM конфігурація з двома рівнями шуму прискорення
    (low для балістики, high для маневру). IMMEstimator розрізняє моделі
    через likelihood, що залежить від P (а P залежить від Q).
    """
    return fx_ballistic(x, dt)


def fx_bounce(x, dt):
    """
    9D модель відскоку від підлоги: інверсія vy із epsilon=0.75,
    тертя vx/vz із friction=0.85. Accel скидається у 0 — відскок це
    детермінований імпульс, що "очищує" попередню історію accel
    (контактна сила реалізується через зміну швидкості, не через
    залишкове прискорення).
    """
    epsilon = 0.75
    friction = 0.85
    x_next = np.copy(x)
    vx_new = x[1] * friction
    vy_new = -x[4] * epsilon
    vz_new = x[7] * friction

    x_next[0] += vx_new * dt
    x_next[3] += vy_new * dt
    x_next[6] += vz_new * dt

    x_next[1] = vx_new
    x_next[4] = vy_new
    x_next[7] = vz_new

    # Скидаємо залишкове прискорення — відскок це "новий старт".
    x_next[2] = 0.0
    x_next[5] = 0.0
    x_next[8] = 0.0

    return x_next


def hx(x):
    """
    Вимірювальна функція 9D → 3D: видобуваємо лише позиції (індекси 0, 3, 6).
    Швидкості (1, 4, 7) та прискорення (2, 5, 8) — не вимірюються, лише
    оцінюються через UKF/IMM.
    """
    return np.array([x[0], x[3], x[6]])

def measurement_transform(prediction_data, K, R_matrix, camera_pos,
                          ball_diameter=0.21):
    """
    Перетворює запис детекції (px, py, ширина рамки) у тривимірний вимір
    [X, Y, Z] світових координат у системі майданчика з конвенцією:
        X — поперек майданчика (по лицевій лінії),
        Y — вертикальна вісь (вгору, перпендикулярно підлозі),
        Z — вздовж довгої сторони (по бічній лінії).
    Підлога відповідає Y = 0; стеля — Y > 0.

    Раніше тут була інверсія `raw_y = -raw_y` — рудимент конвенції
    Z=вертикаль, що робив висоту від'ємною і ламав гравітаційну
    компоненту fx_ballistic. Прибрано після уніфікації осей.
    """
    u_cam, v_cam, w_cam = (prediction_data['x_pos'], prediction_data['y_pos'],
                           prediction_data['w_box'])
    raw_x, raw_y, raw_z = get_3d_position(u_cam, v_cam, w_cam, K,
                                          R_matrix, camera_pos, ball_diameter)
    measurement = np.array([raw_x, raw_y, raw_z], dtype=np.float32)

    return measurement

def get_dynamic_transition_matrix(y, vy,
                                  mahalanobis_sq=0.0,
                                  hit_residual_min_sq=8.0,
                                  hit_residual_max_sq=11.34,
                                  bounce_height_max=0.55,
                                  frames_since_update=0,
                                  bounce_max_coast=3):
    """
    Повертає марковську матрицю 3x3 переходів IMM для моделей
    [Ballistic, Hit, Bounce] на наступний крок передбачення.

    Базова матриця "залипає" на Ballistic (M[0,0]=0.95): без зовнішніх
    тригерів м'яч продовжує летіти по балістичній траєкторії.

    Тригери:

    1) **Bounce** — пасивний геометричний тригер. Якщо м'яч близько до
       підлоги (y < bounce_height_max) і рухається вниз (vy < 0),
       наступний крок різко зміщає Markov-матрицю до Bounce-режиму.
       Поріг підняли 0.3 → 0.55 (2026-05-30): аудит епізоду відбиття
       (кадри 275-290) показав, що ТРЕКОВАНА висота на дні відбиття
       опускається лише до ~0.35 м (raw min ~0.61 м) — стара межа 0.3
       НІКОЛИ не спрацьовувала, тож mu_bounce залишався ~0.01 і реверс
       vy "витягувала" балістична модель за 3 кадри (пізно + рвано).
       0.55 ловить дно відбиття з запасом; умова vy<0 гарантує, що після
       відскоку (vy>0) тригер вимикається й не реверсить швидкість
       повторно.

    2) **Mid-air hit** — тригер за residual'ом. Активується тільки коли
       квадрат Mahalanobis'а останньої прийнятої детекції потрапляє у
       "rare-event" зону. χ²₃-розподіл має 99-перцентиль ≈ 11.34
       (= наш гейт). Зона активації [8.0, 11.34] відповідає верхнім
       ~2% residual'ів — фактично сигнал, що траєкторія раптово
       змінилась (пас/атака/блок), хоч детекція й не настільки
       аберантна, щоб не пройти гейт.

       Раніше пробували min_sq=4.0, але це 25-й хвіст χ²₃ — тригер
       вистрелював у 25% кадрів нормального трекінгу, що подвоїло
       mode_switch_overall_rate_hz (3.46 → 7.18 Hz на тестовому
       сегменті). Підняття порогу до 8.0 повертає mode-switching до
       робочих значень, але зберігає реакцію на справжні удари.

       Параметри:

       - hit_residual_min_sq=8.0:  ~95-й перцентиль χ²₃;
       - hit_residual_max_sq=11.34: верхня межа = поріг гейтингу.

       Поза діапазоном [min, max] інтерполяція клампується (0 нижче,
       1 вище).

    :param y:   вертикальна координата м'яча у світовій СК (м).
    :param vy:  вертикальна швидкість (м/с).
    :param mahalanobis_sq: квадрат Mahalanobis'а останньої прийнятої
        детекції відносно IMM-прогнозу (0 для треків без оновлень).
    :param hit_residual_min_sq, hit_residual_max_sq: діапазон активації
        hit-тригера (квадрати, що відповідають χ²₃-розподілу).
    """
    M = np.array([[0.95, 0.04, 0.01],
                  [0.60, 0.40, 0.00],
                  [0.90, 0.00, 0.10]])

    # Лок 0: сліпий коаст -> розв'язка мод (одинична M). Коли трек довше за
    # bounce_max_coast кадрів іде без вимірювань (м'яч поза кадром / у
    # глибокій оклюзії), ймовірності мод НЕ інформовані даними. М'яч у
    # вільному польоті поза кадром — за визначенням чисто балістичний.
    # Базова матриця має постійний витік Ballistic->Bounce (M[0,2]=0.01);
    # bounce-фільтр сидить на інвертованій vy (~-7 м/с), і IMM
    # interaction-step щокадру вливає ~1% цієї швидкості назад у балістику
    # (~-0.17 м/с/кадр поверх фізики). На довгому підйомі (32 кадри) це
    # з'їдає ~5 м/с і занижує апекс ~10.0 -> ~8.35 м (діагноз tid4,
    # 2026-05-30).
    #
    # Одинична M розриває цей витік: стовпець 0 = [1,0,0] означає, що
    # ЗМІШАНИЙ стартовий стан балістичного фільтра = його ВЛАСНИЙ стан без
    # домішки bounce/hit (omega[i,0]=0 для i!=0). Тобто балістичний фільтр
    # під час коасту еволюціонує чисто фізично. Чому не [[1,0,0]]*3 (як
    # планувалось): там стовпці 1,2 нульові -> cbar[1]=cbar[2]=0 -> filterpy
    # ділить 0/0 у мікшуванні -> NaN у коваріації -> Cholesky падає. У
    # одиничній cbar[j]=mu[j]>0, NaN не виникає. Реакузиція детекції
    # (tsu->0) автоматично повертає повну базову M.
    # Емпірично (tid4, 2026-05-31): одинична матриця дає апекс 8.33->9.50 м.
    # Пробували активне злиття bounce->ballistic (M[2,0]=0.5) щоб прибрати
    # залишкову 1% bounce-домішку з ВИХОДУ — вийшло ГІРШЕ (9.42), бо
    # перехідний re-bleed bounce-стану в балістичний ФІЛЬТР накопичується й
    # несеться до апекса. Кумулятивний витік у фільтр домінує над статичною
    # домішкою у виході, тож нульовий витік (eye) — оптимум. Залишок
    # 9.50 vs ~10.0 (фізика) = 1% bounce у виході; прибрати без витоку в
    # фільтр Марковське мікшування не дозволяє (маса тече через стовпець 0).
    if frames_since_update > bounce_max_coast:
        return np.eye(3)

    # Тригер 1: bounce при контакті з підлогою. Має найвищий пріоритет —
    # після відскоку інформація з residual'у про "удар" не має сенсу.
    # Умова frames_since_update <= bounce_max_coast: тригерити відскок ЛИШЕ
    # коли позиція підкріплена свіжою детекцією. Без цього (аудит 2026-05-30)
    # тригер вистрелював на ФАНТОМНИХ коастинг-траєкторіях: трек, що загубив
    # м'яч, екстраполює вниз під гравітацією, Y падає <0.55, і bounce-режим
    # "відскакує" уявний м'яч (tsu=17-80), хоча реальний м'яч уже за 4-5 м
    # угорі. Реальний відскок має tsu≈0. Це розколювало цілісні треки на
    # уламки (fragmentation 0.235→0.412).
    if y < bounce_height_max and vy < 0 and frames_since_update <= bounce_max_coast:
        M[0] = [0.10, 0.05, 0.85]
        M[1] = [0.10, 0.05, 0.85]
        return M

    # Тригер 2: mid-air hit за residual'ом. Лінійна інтерполяція рядка
    # M[0] у бік M_hit_target. M_hit_target обрано КОНСЕРВАТИВНИМ:
    # навіть при alpha=1 (residual на самому гейті) M[0,0] лишається
    # 0.60 (sticky-ballistic), а Hit-ймовірність зростає лише до 0.35.
    # Сильніший shift ([0.30, 0.65, 0.05], який пробували першою
    # ітерацією) призводив до осциляції Ballistic↔Hit, бо після
    # переходу до Hit residual різко падав (більша Q абсорбує
    # неочікувану швидкість), trigger переставав вистрелювати, IMM
    # повертався у Ballistic, residual знов зростав → нескінченний
    # цикл. Менш агресивний target гасить осциляцію.
    if mahalanobis_sq > hit_residual_min_sq:
        alpha = min(
            1.0,
            (mahalanobis_sq - hit_residual_min_sq)
            / max(hit_residual_max_sq - hit_residual_min_sq, 1e-6),
        )
        M_hit_target = np.array([0.45, 0.50, 0.05])
        M[0] = (1.0 - alpha) * M[0] + alpha * M_hit_target

    return M