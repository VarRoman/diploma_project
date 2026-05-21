import numpy as np
from scipy.optimize import linear_sum_assignment
from IMM_UKF import *
import cv2



def create_imm_estimator(z_initial, dt=0.02):
    frames_per_second = 50
    dt = 1 / frames_per_second
    dim_x = 6
    dim_z = 3
    points = MerweScaledSigmaPoints(n=dim_x, alpha=.1, beta=2., kappa=1.)
    P_init = np.diag([0.1, 50.0, 0.1, 50.0, 0.1, 50.0])
    R_init = np.diag([0.01, 0.01, 0.04])

    # Для балістичної моделі Q мінімальна
    q_var_ballistic = 0.1
    q_b = Q_discrete_white_noise(dim=2, dt=dt, var=q_var_ballistic)
    Q_ballistic = block_diag(q_b, q_b, q_b)

    # Для моделі удару Q велика
    q_var_hit = 100
    q_h = Q_discrete_white_noise(dim=2, dt=dt, var=q_var_hit)
    Q_hit = block_diag(q_h, q_h, q_h)

    # Для відскоку Q середня
    q_var_bounce = 5
    q_bnc = Q_discrete_white_noise(dim=2, dt=dt, var=q_var_bounce)
    Q_bounce = block_diag(q_bnc, q_bnc, q_bnc)

    # Ballistic filter
    ukf_ballistic = UnscentedKalmanFilter(name='ballistic UKF', dim_x=dim_x, dim_z=dim_z, dt=dt, fx=fx_ballistic, hx=hx, points=points)
    ukf_ballistic.P = P_init
    ukf_ballistic.Q = Q_ballistic
    ukf_ballistic.R = R_init

    # Hit filter
    ukf_hit = UnscentedKalmanFilter(name='hit UKF', dim_x=dim_x, dim_z=dim_z, dt=dt, fx=fx_hit, hx=hx, points=points)
    ukf_hit.P = P_init
    ukf_hit.Q = Q_hit
    ukf_hit.R = R_init

    # Bounce filter
    ukf_bounce = UnscentedKalmanFilter(name='bounce UKF', dim_x=dim_x, dim_z=dim_z, dt=dt, fx=fx_bounce, hx=hx, points=points)
    ukf_bounce.P = P_init
    ukf_bounce.Q = Q_bounce
    ukf_bounce.R = R_init

    filters_lt = [ukf_ballistic, ukf_hit, ukf_bounce]
    mu = np.array([0.95, 0.04, 0.01])
    M_base = np.array([[0.95, 0.04, 0.01],
                       [0.60, 0.40, 0.00],
                       [0.90, 0.00, 0.10]])

    imm = IMMEstimator(filters_lt, mu, M_base)

    # Ініціалізуємо стан
    initial_x = np.array([z_initial[0], 0.0, z_initial[1], 0.0, z_initial[2], 0.0])
    for f in imm.filters:
        f.x = initial_x.copy()
    imm._compute_state_estimate()

    return imm

class Track:
    def __init__(self, track_id, z_initial, dt):
        self.track_id = track_id
        self.imm = create_imm_estimator(z_initial, dt)

        # Життєвий цикл
        self.state = 'Tentative'  # Можливі: 'Tentative', 'Confirmed', 'Deleted'
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.h_matrix = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0]
        ], dtype=float)

    def predict(self):
        self.imm.M = get_dynamic_transition_matrix(self.imm.x[2], self.imm.x[3])
        self.imm.predict()
        self.time_since_update += 1

    def update(self, z):
        self.imm.update(z)
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        if self.state == 'Tentative' and self.hits >= 3:
            self.state = 'Confirmed'

    def mark_missed(self):
        self.hit_streak = 0
        self.imm.update(None)  # Екстраполяція без вимірювання

    def get_mahalanobis_distance(self, z, R_matrix):
        """
        Обчислює відстань Махаланобіса між прогнозом IMM та новою детекцією.
        """
        # Проектуємо змішаний стан x_prior та коваріацію P_prior у простір вимірювань
        z_mean = np.dot(self.h_matrix, self.imm.x_prior)
        S = np.dot(self.h_matrix, np.dot(self.imm.P_prior, self.h_matrix.T)) + R_matrix

        y = z - z_mean  # Вектор невязки

        try:
            S_inv = np.linalg.inv(S)
            dist_sq = np.dot(y.T, np.dot(S_inv, y))
            return dist_sq
        except np.linalg.LinAlgError:
            return 1e5


class IMMTracker:
    def __init__(self, dt=0.02, max_age=50, min_hits=3, gating_threshold=11.34):
        self.dt = dt
        self.max_age = max_age  # К-сть кадрів до видалення треку (50 кадрів = 1 сек)
        self.min_hits = min_hits  # К-сть кадрів для підтвердження
        self.gating_threshold = gating_threshold  # Хі-квадрат поріг для 3 ступенів свободи (99%)

        self.tracks = []
        self.next_id = 1
        self.R_matrix = np.diag([0.01, 0.01, 0.04])  # Базова матриця похибки

    def update(self, detections_3d):
        """
        detections_3d: список масивів np.array([x, y, z]) для всіх знайдених об'єктів у кадрі
        """
        for track in self.tracks:
            track.predict()

        if len(detections_3d) == 0:
            for track in self.tracks:
                track.mark_missed()
            self._manage_lifecycle()
            return

        if len(self.tracks) == 0:
            for z in detections_3d:
                self.tracks.append(Track(self.next_id, z, self.dt))
                self.next_id += 1
            return

        cost_matrix = np.full((len(self.tracks), len(detections_3d)), 1e5)

        for t, track in enumerate(self.tracks):
            for d, z in enumerate(detections_3d):
                dist_sq = track.get_mahalanobis_distance(z, self.R_matrix)
                if dist_sq < self.gating_threshold:
                    cost_matrix[t, d] = dist_sq

        # 3. Асоціація (Угорський алгоритм)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_detections = set(range(len(detections_3d)))

        # 4. Оновлення знайдених пар
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < self.gating_threshold:
                self.tracks[r].update(detections_3d[c])
                unmatched_tracks.remove(r)
                unmatched_detections.remove(c)

        # 5. Обробка пропущених треків
        for t in unmatched_tracks:
            self.tracks[t].mark_missed()

        # 6. Створення нових треків для нерозпізнаних детекцій
        for d in unmatched_detections:
            self.tracks.append(Track(self.next_id, detections_3d[d], self.dt))
            self.next_id += 1

        # 7. Видалення старих треків
        self._manage_lifecycle()

    def _manage_lifecycle(self):
        active_tracks = []
        for track in self.tracks:
            # Видаляємо трек, якщо він довго не оновлювався
            if track.time_since_update > self.max_age:
                track.state = 'Deleted'

            # Трек залишається, якщо він не видалений
            if track.state != 'Deleted':
                active_tracks.append(track)

        self.tracks = active_tracks

    def get_confirmed_tracks(self):
        """Повертає позиції лише тих треків, в яких ми впевнені"""
        return [t for t in self.tracks if t.state == 'Confirmed']