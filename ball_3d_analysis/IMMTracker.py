import numpy as np
from scipy.optimize import linear_sum_assignment
from IMM_UKF import *
import cv2



def create_imm_estimator(z_initial, dt=0.02, v_initial=None):
    """
    Створює IMM-естиматор з трьох UKF (ballistic / hit / bounce).

    :param z_initial: 3-вектор світової позиції м'яча [x, y, z] при ініціалізації.
    :param dt:        крок часу (= 1 / FPS).
    :param v_initial: опціональний 3-вектор початкових швидкостей [vx, vy, vz]
                      у м/с. Якщо None — швидкості ініціалізуються нулями.
                      Передавайте оцінку (z_k − z_{k−1}) / Δt, коли можемо
                      пов'язати поточну детекцію з найближчою попередньою
                      незасоційованою детекцією — це різко скорочує час
                      сходження UKF-фільтрів і запобігає катастрофічним
                      сплескам у |residual|/Mahalanobis у перші кадри
                      життя нового треку.
    """
    # dt is forwarded from IMMTracker (1 / source FPS);
    # default corresponds to 50 FPS
    dim_x = 6
    dim_z = 3
    points = MerweScaledSigmaPoints(n=dim_x, alpha=.1, beta=2., kappa=1.)
    # Якщо швидкість ініціалізована з фінітної різниці двох детекцій, її
    # σ становить ≈ √2·σ_R/Δt (від ~7 м/с до ~14 м/с при дефолтних R, dt).
    # Тому коваріація швидкостей лишається 50 — це консервативний пріор,
    # що дозволяє фільтру швидко поглинути реальну швидкість.
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
    ukf_ballistic = UnscentedKalmanFilter(name='ballistic UKF', dim_x=dim_x,
                    dim_z=dim_z, dt=dt, fx=fx_ballistic, hx=hx, points=points)
    ukf_ballistic.P = P_init
    ukf_ballistic.Q = Q_ballistic
    ukf_ballistic.R = R_init

    # Hit filter
    ukf_hit = UnscentedKalmanFilter(name='hit UKF', dim_x=dim_x, dim_z=dim_z,
                                    dt=dt, fx=fx_hit, hx=hx, points=points)
    ukf_hit.P = P_init
    ukf_hit.Q = Q_hit
    ukf_hit.R = R_init

    # Bounce filter
    ukf_bounce = UnscentedKalmanFilter(name='bounce UKF', dim_x=dim_x,
                    dim_z=dim_z, dt=dt, fx=fx_bounce, hx=hx, points=points)
    ukf_bounce.P = P_init
    ukf_bounce.Q = Q_bounce
    ukf_bounce.R = R_init

    filters_lt = [ukf_ballistic, ukf_hit, ukf_bounce]
    mu = np.array([0.95, 0.04, 0.01])
    M_base = np.array([[0.95, 0.04, 0.01],
                       [0.60, 0.40, 0.00],
                       [0.90, 0.00, 0.10]])

    imm = IMMEstimator(filters_lt, mu, M_base)

    # Ініціалізуємо стан [x, vx, y, vy, z, vz]. Якщо передано v_initial —
    # використовуємо його; інакше — нулі (legacy-поведінка).
    if v_initial is None:
        vx0 = vy0 = vz0 = 0.0
    else:
        vx0, vy0, vz0 = (float(v_initial[0]), float(v_initial[1]),
                         float(v_initial[2]))
    initial_x = np.array([z_initial[0], vx0,
                          z_initial[1], vy0,
                          z_initial[2], vz0])
    for f in imm.filters:
        f.x = initial_x.copy()
    imm._compute_state_estimate()

    return imm

class Track:
    # Санітарне обмеження на bootstrap-швидкість (м/с): якщо обчислена з
    # фінітної різниці швидкість перевищує цей поріг — bootstrap НЕ
    # застосовуємо, бо це майже напевно з'єднання двох різних об'єктів.
    BOOTSTRAP_MAX_SPEED = 35.0

    def __init__(self, track_id, z_initial, dt, v_initial=None,
                 initial_hits=1, min_hits=3):
        """
        :param v_initial:   опціональний 3-вектор початкових швидкостей.
                            Якщо передано — UKF-фільтри стартують зі
                            «справжнім» вектором швидкості замість нулів.
        :param initial_hits: к-сть hits, з якою стартує трек. Для
                            bootstrap'нутих треків розумно ставити 2
                            (бо ми фактично спостерігали дві послідовні
                            детекції — попередню «незасоційовану» та
                            поточну). Дозволяє швидше промувати трек до
                            'Confirmed', не чекаючи зайвого кадру.
        :param min_hits:    к-сть оновлень (hits), необхідна для переходу
                            трек у стан 'Confirmed'. Раніше було
                            захардкоджено 3 — тепер прокидається з
                            IMMTracker (CLI), щоб параметр з sweep'а
                            справді впливав на трек-стейт-машину.
        """
        self.track_id = track_id
        self.dt = float(dt)
        self.imm = create_imm_estimator(z_initial, dt, v_initial=v_initial)

        # Запам'ятовуємо ПЕРШУ детекцію — використовується для
        # внутрішньо-трекного bootstrap'у швидкості на другому хіті.
        # Якщо v_initial вже передано ззовні (Mechanism B з IMMTracker),
        # повторний bootstrap пропускаємо.
        self.z_initial = np.asarray(z_initial, dtype=float).copy()
        self._velocity_bootstrapped = (v_initial is not None)

        # Життєвий цикл
        self.state = 'Tentative'  # Можливі: 'Tentative', 'Confirmed', 'Deleted'
        self.time_since_update = 0
        self.hits = int(initial_hits)
        self.hit_streak = int(initial_hits)
        self.min_hits = int(min_hits)
        # Останній прийнятий Mahalanobis^2 (gating-residual) — передається
        # у get_dynamic_transition_matrix для активації hit-тригера у
        # наступному predict(). Скидається у 0 після кадрів без оновлень.
        self._last_mahalanobis_sq = 0.0
        self.h_matrix = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0]
        ], dtype=float)

    def predict(self):
        self.imm.M = get_dynamic_transition_matrix(
            self.imm.x[2], self.imm.x[3],
            mahalanobis_sq=self._last_mahalanobis_sq,
        )
        self.imm.predict()
        self.time_since_update += 1

    def update(self, z):
        # Mechanism A — Intra-track velocity bootstrap.
        # На переході hits=1 → 2 (тобто коли трек уперше отримує
        # ДРУГУ детекцію після створення) UKF-фільтри ще мали v=0 у
        # пріорі поточного кадру. Це призводить до:
        #   1) несхожої позиції-прогнозу (за реальної v>0 ball дрейфує
        #      від прогнозу) → велике |residual|, велика Mahalanobis;
        #   2) повільного «зсуву» Kalman-gain'ом швидкості до правди
        #      (через велику σ_v=√50≈7 м/с, K_v нікчемно мала за 1 крок).
        # Виправляємо це примусово: ОБРАХОВУЄМО v_bootstrap з фінітної
        # різниці (z − z_initial) / (gap·dt) і ВПИСУЄМО її у швидкісні
        # компоненти кожного фільтра до того, як викличемо imm.update.
        # Так predict-update проходять із коректним пріором швидкості з
        # САМОГО ПЕРШОГО використання, а Mahalanobis на 3-му кадрі вже
        # дорівнює 0–1 замість 3–4.
        do_bootstrap = (
            z is not None
            and self.hits == 1
            and not self._velocity_bootstrapped
            and self.time_since_update > 0
        )
        if do_bootstrap:
            gap_dt = self.time_since_update * self.dt
            z_arr = np.asarray(z, dtype=float)
            v_boot = (z_arr - self.z_initial) / gap_dt
            speed = float(np.linalg.norm(v_boot))
            if 0.0 < speed <= self.BOOTSTRAP_MAX_SPEED:
                # Вписуємо швидкість у ВСІ внутрішні UKF-фільтри (вони
                # роздільні; mixed-стан перерахується автоматично у
                # imm.update → _compute_state_estimate).
                # Так само переписуємо x_prior, бо саме він
                # використовуватиметься у sigma-точках наступного
                # predict.
                for f in self.imm.filters:
                    f.x[1] = v_boot[0]
                    f.x[3] = v_boot[1]
                    f.x[5] = v_boot[2]
                    f.x_prior[1] = v_boot[0]
                    f.x_prior[3] = v_boot[1]
                    f.x_prior[5] = v_boot[2]
                # mixed state теж оновлюємо.
                self.imm.x[1] = v_boot[0]
                self.imm.x[3] = v_boot[1]
                self.imm.x[5] = v_boot[2]
                self.imm.x_prior[1] = v_boot[0]
                self.imm.x_prior[3] = v_boot[1]
                self.imm.x_prior[5] = v_boot[2]
                self._velocity_bootstrapped = True

        self.imm.update(z)
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        if self.state == 'Tentative' and self.hits >= self.min_hits:
            self.state = 'Confirmed'

    def mark_missed(self):
        self.hit_streak = 0
        # На кадрах без вимірювання residual не визначений — скидаємо у 0,
        # щоб у наступному predict() hit-тригер не активувався фантомно
        # з застарілих даних.
        self._last_mahalanobis_sq = 0.0
        self.imm.update(None)  # Екстраполяція без вимірювання

    def get_mahalanobis_distance(self, z, R_matrix):
        """
        Обчислює відстань Махаланобіса між прогнозом IMM та
        новою детекцією.
        """
        # Проектуємо змішаний стан x_prior та коваріацію
        # P_prior у простір вимірювань
        z_mean = np.dot(self.h_matrix, self.imm.x_prior)
        S = np.dot(self.h_matrix, np.dot(self.imm.P_prior,
                        self.h_matrix.T)) + R_matrix

        y = z - z_mean  # Вектор невязки

        try:
            S_inv = np.linalg.inv(S)
            dist_sq = np.dot(y.T, np.dot(S_inv, y))
            return dist_sq
        except np.linalg.LinAlgError:
            return 1e5


class IMMTracker:
    def __init__(self, dt=0.02, max_age=150, min_hits=3, gating_threshold=11.34,
                 bootstrap_max_gap_frames=5, bootstrap_max_dist=2.5,
                 bootstrap_max_speed=35.0):
        """
        :param dt:                    крок часу (= 1 / FPS).
        :param max_age:               к-сть кадрів без оновлення до
                                      видалення треку. Default 150 (= 3 с
                                      при 50 FPS) — пiдiбраний sweep'ом
                                      (logs/sweep_v1): на тестовому
                                      сегменті дає lifeP90=325.6 кадрiв
                                      (vs 214.8 для max_age=100 i
                                      117.2 для legacy 50), при тiй
                                      самiй фрагментацiї.
        :param min_hits:              к-сть кадрів для підтвердження
                                      треку (Tentative → Confirmed).
        :param gating_threshold:      χ²-поріг гейтингу Махаланобіса
                                      (df=3, p=0.99 → 11.34).
        :param bootstrap_max_gap_frames: максимальний розрив (кадри) між
                                      пендінговою детекцією та поточною,
                                      щоб виконати bootstrap швидкості.
                                      Default 5 (= 0.1 с при 50 FPS).
        :param bootstrap_max_dist:    максимальна 3D-відстань (м) між
                                      пендінговою та поточною детекцією
                                      для bootstrap. Default 2.5 м —
                                      пасує для м'яча, що рухається до
                                      ~25 м/с, при розривах ≤ 0.1 с.
        :param bootstrap_max_speed:   санітарне обмеження на
                                      bootstrap-швидкість (м/с). Якщо
                                      обчислена |v| > цього порогу —
                                      не використовуємо bootstrap
                                      (швидше за все фальшиве з'єднання
                                      двох різних об'єктів). Default 35
                                      м/с — приблизна верхня межа подачі
                                      елітного м'яча у волейболі.
        """
        self.dt = dt
        self.max_age = max_age
        self.min_hits = min_hits
        self.gating_threshold = gating_threshold

        self.bootstrap_max_gap_frames = int(bootstrap_max_gap_frames)
        self.bootstrap_max_dist = float(bootstrap_max_dist)
        self.bootstrap_max_speed = float(bootstrap_max_speed)

        self.tracks = []
        self.next_id = 1
        self.R_matrix = np.diag([0.01, 0.01, 0.04])  # Базова матриця похибки

        # Ring-буфер невідповідних (unassigned) детекцій з кількох останніх
        # кадрів. Використовується для bootstrap'у швидкості при створенні
        # нового треку: якщо поточна незасоційована детекція близька до
        # детекції з останніх N кадрів, ініціалізуємо новий трек з
        # v = (z_now − z_prev) / (Δfr · dt) замість нулів.
        self._spawn_buffer = []  # елементи: (np.array z, int frame_idx)
        self._frame_counter = 0

    def update(self, detections_3d):
        """
        detections_3d: список масивів np.array([x, y, z]) для
        всіх знайдених об'єктів у кадрі
        """
        self._frame_counter += 1

        for track in self.tracks:
            track.predict()

        if len(detections_3d) == 0:
            for track in self.tracks:
                track.mark_missed()
            self._age_spawn_buffer()
            self._manage_lifecycle()
            return

        if len(self.tracks) == 0:
            # Особливий випадок: треків ще немає. Будь-яка детекція має
            # шанс на bootstrap, якщо у spawn-буфері є попередник.
            for z in detections_3d:
                self._spawn_track_from_unmatched(z)
            self._age_spawn_buffer()
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
                # Зберігаємо Mahalanobis^2 НА МОМЕНТ accept'у — він піде
                # у get_dynamic_transition_matrix наступного predict().
                self.tracks[r]._last_mahalanobis_sq = float(
                    cost_matrix[r, c]
                )
                self.tracks[r].update(detections_3d[c])
                unmatched_tracks.remove(r)
                unmatched_detections.remove(c)

        # 5. Обробка пропущених треків
        for t in unmatched_tracks:
            self.tracks[t].mark_missed()

        # 6. Створення нових треків для нерозпізнаних детекцій
        #    (із bootstrap'ом швидкості, якщо знайдено попередника).
        for d in unmatched_detections:
            self._spawn_track_from_unmatched(detections_3d[d])

        # 7. Старіння spawn-буфера + видалення старих треків
        self._age_spawn_buffer()
        self._manage_lifecycle()

    # ------------------------------------------------------------------
    # Допоміжні методи bootstrap'у початкової швидкості
    # ------------------------------------------------------------------
    def _spawn_track_from_unmatched(self, z_now):
        """
        Створює новий Track для незасоційованої детекції z_now. Якщо у
        spawn-буфері є точка-попередник, що задовольняє обмеженням
        (близько у просторі, недавно у часі, узгоджена швидкість) —
        ініціалізує трек з v_initial = (z_now − z_prev)/(Δfr · dt) та
        hits=2; інакше — звичайна ініціалізація з нулевою швидкістю.
        Spawn-точка, яку використано для bootstrap'у, видаляється з
        буфера (один-до-одного), щоб уникнути «склейки» двох треків з
        однієї історичної детекції.
        """
        z_now_arr = np.asarray(z_now, dtype=float)
        best_idx = -1
        best_dist = self.bootstrap_max_dist
        best_v = None
        best_gap = 0

        for i, (z_prev, f_prev) in enumerate(self._spawn_buffer):
            gap = self._frame_counter - f_prev
            if gap <= 0 or gap > self.bootstrap_max_gap_frames:
                continue
            d_xyz = float(np.linalg.norm(z_now_arr - z_prev))
            if d_xyz >= best_dist:
                continue
            gap_dt = gap * self.dt
            if gap_dt <= 0:
                continue
            v_cand = (z_now_arr - z_prev) / gap_dt
            speed = float(np.linalg.norm(v_cand))
            if speed > self.bootstrap_max_speed:
                # Швидкість за межами фізично можливої — не bootstrap'имо.
                continue
            best_idx = i
            best_dist = d_xyz
            best_v = v_cand
            best_gap = gap

        if best_idx >= 0:
            # Bootstrap! Створюємо трек з ненульовою v та hits=2.
            self.tracks.append(
                Track(self.next_id, z_now_arr, self.dt,
                      v_initial=best_v, initial_hits=2,
                      min_hits=self.min_hits)
            )
            self.next_id += 1
            # Видаляємо використану spawn-точку.
            del self._spawn_buffer[best_idx]
        else:
            # Класичний spawn з нульовою швидкістю + кладемо детекцію
            # у буфер для майбутнього bootstrap'у.
            self.tracks.append(
                Track(self.next_id, z_now_arr, self.dt,
                      min_hits=self.min_hits)
            )
            self.next_id += 1
            self._spawn_buffer.append(
                (z_now_arr.copy(), self._frame_counter)
            )

    def _age_spawn_buffer(self):
        """Видаляє з буфера записи, старші за bootstrap_max_gap_frames."""
        cutoff = self._frame_counter - self.bootstrap_max_gap_frames
        self._spawn_buffer = [
            (z, f) for (z, f) in self._spawn_buffer if f > cutoff
        ]

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