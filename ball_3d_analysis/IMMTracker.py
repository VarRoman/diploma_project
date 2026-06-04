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
    # default corresponds to 50 FPS.
    # Step 2.B (Phase 3): 6D CV-state → 9D CA-state.
    # state = [x, vx, ax, y, vy, ay, z, vz, az]
    dim_x = 9
    dim_z = 3
    points = MerweScaledSigmaPoints(n=dim_x, alpha=.1, beta=2., kappa=1.)
    # P_init для 9D стану. Накачуємо коваріацію accel ~25 (м/с²)²
    # (σ_a ≈ 5 м/с²) — це дозволяє фільтру швидко поглинути перший
    # ненульовий accel із вимірювань, не вибухнувши гейтом одразу.
    # Якщо швидкість ініціалізована з фінітної різниці двох детекцій, її
    # σ становить ≈ √2·σ_R/Δt (від ~7 м/с до ~14 м/с при дефолтних R, dt),
    # тому коваріація швидкостей лишається 50.
    P_init = np.diag([
        0.1, 50.0, 25.0,
        0.1, 50.0, 25.0,
        0.1, 50.0, 25.0,
    ])
    # R: коваріація шуму вимірювання у світових координатах [x, y, z] (м²).
    # σ_z ≈ 0.7 м (моно-камерна глибина має таку похибку через
    # Z_c = f·D / bbox_w). Користувач підтвердив, що ці значення дають
    # приємну плавність — спроба зменшити σ_z до 0.5 м (для зменшення
    # відставання) робила траєкторію надто рваною.
    R_init = np.diag([0.02, 0.02, 0.5])

    # Q-матриці для 9D CA-state. У Q_discrete_white_noise(dim=3, ...)
    # параметр var має семантику "дисперсія шуму джерку" — нижня
    # права клітинка блоку = var, що задає, наскільки accel може
    # випадково "крутитися" за крок.
    #
    # ballistic: var=0.1 — м'яч у вільному польоті фактично має accel = g
    # + drag (smooth), залишковий accel-state ≈ 0; шум джерку маленький.
    q_var_ballistic = 0.1
    q_b = Q_discrete_white_noise(dim=3, dt=dt, var=q_var_ballistic)
    Q_ballistic = block_diag(q_b, q_b, q_b)

    # hit: var=400 — це σ_jerk ≈ 20 (м²/с^6)^{1/2}, що дозволяє accel-стейту
    # "прокинутись" від ~0 до десятків м/с² за один-два кадри. Тобто
    # імпульсний удар реалізується НЕ через стрибок швидкості напряму,
    # а через ріст accel-стейту, який потім інтегрується у швидкість.
    # Це структурне моделювання маневру (CV-CA IMM Singer-style), на
    # відміну від наївного збільшення Q[v], що ламало гейтинг у 6D-версії.
    q_var_hit = 400
    q_h = Q_discrete_white_noise(dim=3, dt=dt, var=q_var_hit)
    Q_hit = block_diag(q_h, q_h, q_h)

    # bounce: var=5 — помірний шум джерку. Після контакту з підлогою
    # fx_bounce уже скидає accel у 0; Q[a] лише страхує від raptових
    # відхилень у наступних кадрах.
    q_var_bounce = 5
    q_bnc = Q_discrete_white_noise(dim=3, dt=dt, var=q_var_bounce)
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

    # Ініціалізуємо 9D стан [x, vx, ax, y, vy, ay, z, vz, az].
    # Якщо передано v_initial — використовуємо його; інакше — нулі.
    # accel-компоненти завжди стартують з 0: ми не знаємо, чи м'яч
    # був у момент ініціалізації під дією impulse'у; пріор accel=0
    # відповідає чистій балістиці. Високий P_init[a]=25 дозволяє
    # фільтру швидко "виявити" ненульовий accel при потребі.
    if v_initial is None:
        vx0 = vy0 = vz0 = 0.0
    else:
        vx0, vy0, vz0 = (float(v_initial[0]), float(v_initial[1]),
                         float(v_initial[2]))
    initial_x = np.array([
        z_initial[0], vx0, 0.0,
        z_initial[1], vy0, 0.0,
        z_initial[2], vz0, 0.0,
    ])
    for f in imm.filters:
        f.x = initial_x.copy()
        # filterpy.IMMEstimator у конструкторі копіює f.x_post у imm.x_post
        # ОДИН РАЗ — і це відбувається ДО того, як ми перезаписали f.x.
        # Якщо не перевизначити f.x_post тут вручну, у першому snapshot треку
        # (на кадрі-народжування, ДО першого update'у) у JSONL потрапляє
        # zeros(6) → live_view/overlay малюють точку у світовому (0,0,0) =
        # лівий-передній кут майданчика → видима "лінія з кута" для кожного
        # нового треку. Поле x_post не впливає на predict/update — це лише
        # post-step snapshot для логування — тому правка безпечна.
        f.x_post = initial_x.copy()
    imm._compute_state_estimate()
    # Те саме на рівні IMM-агрегації: imm.x_post читається у snapshot_track
    # з run_segment.py:214. Перевизначаємо тут до повернення.
    imm.x_post = initial_x.copy()

    return imm

class Track:
    # Санітарне обмеження на bootstrap-швидкість (м/с): якщо обчислена з
    # фінітної різниці швидкість перевищує цей поріг — bootstrap НЕ
    # застосовуємо, бо це майже напевно з'єднання двох різних об'єктів.
    BOOTSTRAP_MAX_SPEED = 35.0

    # Step 2.A (Phase 3): track-level covariance reset on radical residual.
    # Якщо поточний Mahalanobis^2 високий (хвіст ~5% χ²₃), вважаємо, що
    # м'яч щойно зробив імпульсний маневр (удар, блок) — і "відкриваємо"
    # коваріацію швидкості та прискорення ПЕРЕД викликом imm.update(z).
    # Більший P[v, a] = більший Kalman gain → новий вимір швидко
    # "втягує" state у нову траєкторію без багатокадрового лагу.
    # Без цього: після удару фільтр бачить residual ≈ 1 м (z прискорився
    # геометрично), повільно зменшує його за 3-5 кадрів, накопичує
    # фантомні mah > гейту, і час від часу втрачає трек.
    COV_RESET_MAH_SQ_THRESHOLD = 8.0   # ~95-й перцентиль χ²₃
    COV_RESET_INFLATE_V = 50.0         # (м/с)² — той самий рівень, що P_init[v]
    COV_RESET_INFLATE_A = 100.0        # (м/с²)² — 4× P_init[a], сильна "відкритість"

    # Phase B: Delayed-accept hysteresis.
    # Підозріла детекція (mah_sq > THRESHOLD) не приймається одразу — вона
    # "заморожується" у _pending і чекає підтвердження з наступних кадрів.
    # Якщо підтвердження є (наступна детекція теж далеко) — маневр реальний.
    # Якщо наступна детекція повернулась до норми — pending був FP, скидається.
    HYSTERESIS_MAH_SQ_THRESHOLD = 8.0   # поріг "підозри" (збігається з COV_RESET)
    HYSTERESIS_CONFIRM_THRESHOLD = 5.0  # mah_sq наступної < цього → pending є FP
    HYSTERESIS_WINDOW = 6               # максимум кадрів очікування підтвердження
    COAST_THRESHOLD = 4                 # пропусків поспіль = "коастинг": висока
                                        # mah_sq очікувана (P зросла природно),
                                        # а не симптом маневру — не inflate, не defer

    # Coast covariance cap. Під час коастингу P росте необмежено (вимір.
    # дані: P_vel 2.5→700, P_pos 0→243, P_acc→1330 за 57 кадрів). При
    # P_vel~300-700 sigma-точки UKF розповзаються на ±17-26 м/с, і
    # КВАДРАТИЧНИЙ drag (a=-k·|v|·v у fx_ballistic) через нерівність Єнсена
    # тягне sigma-mean швидкості до нуля, ПЕРЕБИВАЮЧИ лінійний gravity-term
    # (-g) → vy замерзає (~-1 м/с) замість розгону вниз → трек «висить»
    # замість балістичної дуги (tid4 завис на y=8 м усі 80 кадрів коастингу
    # замість падіння під гравітацією). Кап тримає sigma-точки тісними →
    # gravity знову домінує → чиста парабола поза кадром. Капи на рівні
    # P_init (vel) / cov-reset (acc); pos=9 (±3 м) узгоджено з Gate A
    # residual-стелажем, тримає Mahalanobis-гейт осмисленим на коастингу.
    COAST_P_POS_CAP = 9.0    # (м²) — ±3 м, = Gate A max_assoc_residual
    COAST_P_VEL_CAP = 50.0   # (м/с)² — = P_init[v]
    COAST_P_ACC_CAP = 100.0  # (м/с²)² — = COV_RESET_INFLATE_A

    def __init__(self, track_id, z_initial, dt, v_initial=None,
                 initial_hits=1, min_hits=3, enable_hysteresis=False,
                 enable_cov_reset=False, enable_coast_cov_cap=False,
                 enable_coast_z_fence=False, coast_z_min=-3.0, coast_z_max=21.0,
                 hit_residual_min_sq=8.0, hit_residual_max_sq=11.34,
                 bounce_height_max=0.55, bounce_max_coast=3,
                 m_hit_target=None):
        """
        :param enable_cov_reset: вмикає Step 2.A track-level covariance reset
                            (inflate P[v,a] при mah_sq>поріг). Default False
                            (Path B). Вимкнення лишає residual "пожити" 1-2
                            кадри після маневру → hit-тригер встигає
                            вистрілити і IMM-likelihood підіймає μ_hit.
                            Увімкнений cov-reset снапить маневр за 1 кадр і
                            короутить IMM-режими (cov-reset і hit —
                            субститути за один і той самий residual-сигнал).
        :param enable_hysteresis: вмикає Phase B delayed-accept hysteresis
                            (відкладання підозрілих детекцій у _pending).
                            Default False: на тест-даних гістерезис майже
                            не вмикається, АЛЕ там, де вмикається, він
                            канібалізує cov-reset (Step 2.A) та hit-тригер —
                            усі три ділять поріг mah_sq=8, і defer обнуляє
                            _last_mahalanobis_sq, тож наступний predict не
                            бачить residual'у для hit-режиму. Тримаємо OFF,
                            доки не доналаштуємо cov-reset+hit; coast-skip
                            (через _missed_streak) лишається активним
                            незалежно від прапорця — це обробка оклюзії,
                            а не defer.
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
        self.enable_hysteresis = bool(enable_hysteresis)
        self.enable_cov_reset = bool(enable_cov_reset)
        self.enable_coast_cov_cap = bool(enable_coast_cov_cap)
        # Coast Z-fence (2026-06-04): під час коастингу немає вимірювання, тож
        # жоден гейт не діє — predict() екстраполює світову глибину Z (індекс 6)
        # за контамінованою depth-швидкістю vz (індекс 7). На даних: vz доростає
        # до ±19 м/с (бо біля межі видимої області w_box обвалюється → Z=f·D/w
        # роздувається), і коаст несе м'яч у Z=33-42 м — фізично неможливо на
        # майданчику 18 м. Фенс клампить Z прогнозу у [coast_z_min, coast_z_max]
        # і гасить vz/az, які штовхають за фенс. Якщо м'яч справді покинув корт —
        # max_age його видалить; доки коастить у межах — глибина лишається
        # фізичною. Діє ЛИШЕ під час коасту (mark_missed), не чіпає update-кадри.
        self.enable_coast_z_fence = bool(enable_coast_z_fence)
        self.coast_z_min = float(coast_z_min)
        self.coast_z_max = float(coast_z_max)
        # IMM-тригери (Phase F експеримент): прокидаються у
        # get_dynamic_transition_matrix кожного predict(). Дефолти збігаються
        # з сигнатурою функції — A/B-сумісність зі старими прогонами.
        self.hit_residual_min_sq = float(hit_residual_min_sq)
        self.hit_residual_max_sq = float(hit_residual_max_sq)
        self.bounce_height_max = float(bounce_height_max)
        self.bounce_max_coast = int(bounce_max_coast)
        self.m_hit_target = (None if m_hit_target is None
                             else np.asarray(m_hit_target, dtype=float))
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
        # Phase B: hysteresis state
        self._pending = None               # Optional[Tuple[np.ndarray, float]]: (z, mah_sq)
        self._pending_frames_elapsed = 0  # кадрів з моменту встановлення pending
        self._missed_streak = 0           # послідовних mark_missed (для coast detection)
        # 9D h_matrix: позиційні компоненти — індекси 0, 3, 6 у state
        # [x, vx, ax, y, vy, ay, z, vz, az]. Швидкості та прискорення
        # не вимірюються напряму, лише оцінюються через UKF.
        self.h_matrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],   # x
            [0, 0, 0, 1, 0, 0, 0, 0, 0],   # y
            [0, 0, 0, 0, 0, 0, 1, 0, 0],   # z
        ], dtype=float)

    def _inflate_covariance(self):
        """Inflate P[v,a] для всіх sub-фільтрів та IMM-агрегації."""
        for f in self.imm.filters:
            f.P[1, 1] += self.COV_RESET_INFLATE_V
            f.P[4, 4] += self.COV_RESET_INFLATE_V
            f.P[7, 7] += self.COV_RESET_INFLATE_V
            f.P[2, 2] += self.COV_RESET_INFLATE_A
            f.P[5, 5] += self.COV_RESET_INFLATE_A
            f.P[8, 8] += self.COV_RESET_INFLATE_A
        self.imm.P[1, 1] += self.COV_RESET_INFLATE_V
        self.imm.P[4, 4] += self.COV_RESET_INFLATE_V
        self.imm.P[7, 7] += self.COV_RESET_INFLATE_V
        self.imm.P[2, 2] += self.COV_RESET_INFLATE_A
        self.imm.P[5, 5] += self.COV_RESET_INFLATE_A
        self.imm.P[8, 8] += self.COV_RESET_INFLATE_A

    def _collapse_accel_to_ballistic(self):
        """Занулює залишкове прискорення (індекси 2, 5, 8) у всіх станах.

        Викликається при коастингу (mark_missed). Залишковий accel у 9D
        CA-стані валідний лише поки його підтверджують виміри: без виміру
        predict() інтегрує застаріле ax у швидкість щокадру (vx += ax·dt) →
        трек РОЗГАНЯЄТЬСЯ і летить через увесь корт (аномалія кадрів
        611-650: vX тікав −3.2 → −9.7 за 33 кадри оклюзії, X 7.2 → 2.4).
        Фізично м'яч між контактами має нульовий залишковий accel — лише
        g + drag, які вже driving-terms у fx_*. Колапс до балістики лишає
        коастинг constant-velocity замість нестабільної CA-екстраполяції.
        Швидкість/позиція не чіпаються (вони фізично виправдані), P також.
        """
        for f in self.imm.filters:
            f.x[2] = f.x[5] = f.x[8] = 0.0
            f.x_post[2] = f.x_post[5] = f.x_post[8] = 0.0
        self.imm.x[2] = self.imm.x[5] = self.imm.x[8] = 0.0
        self.imm.x_post[2] = self.imm.x_post[5] = self.imm.x_post[8] = 0.0

    @staticmethod
    def _cap_P_matrix(P, caps):
        """PD-безпечний кап діагоналі P до caps[i]. Для кожного i з
        P[i,i] > caps[i] масштабуємо рядок І стовпець i на s=√(cap/P[i,i]).
        Це конгруентне перетворення D·P·D (D=diag, d_i=s), що ЗБЕРІГАЄ
        додатну визначеність (на відміну від наївного P[i,i]=cap, яке при
        великих позадіагональних кореляціях ламає PD → Cholesky-збій у
        sigma-точках UKF). Off-діагоналі масштабуються пропорційно, тож
        кореляційна структура лишається валідною."""
        for i in range(P.shape[0]):
            if P[i, i] > caps[i]:
                s = np.sqrt(caps[i] / P[i, i])
                P[i, :] *= s
                P[:, i] *= s

    def _cap_coast_covariance(self):
        """Обмежує ріст P-діагоналі під час коастингу (див. COAST_P_*_CAP).
        Без капу P росте необмежено → sigma-точки UKF розповзаються →
        квадратичний drag глушить gravity у sigma-mean → vy замерзає, трек
        «висить» замість балістичної дуги. Кап тримає sigma тісними →
        чиста парабола. Доповнює Gate A (обидва б'ють runaway P)."""
        caps = np.array([
            self.COAST_P_POS_CAP, self.COAST_P_VEL_CAP, self.COAST_P_ACC_CAP,
            self.COAST_P_POS_CAP, self.COAST_P_VEL_CAP, self.COAST_P_ACC_CAP,
            self.COAST_P_POS_CAP, self.COAST_P_VEL_CAP, self.COAST_P_ACC_CAP,
        ])
        for f in self.imm.filters:
            self._cap_P_matrix(f.P, caps)
        self._cap_P_matrix(self.imm.P, caps)

    def _fence_coast_depth(self):
        """Клампить світову глибину Z (індекс 6) під час коасту в межі
        [coast_z_min, coast_z_max] і гасить depth-швидкість/прискорення
        (vz=7, az=8), що штовхають за фенс. Застосовується до ВСІХ внутрішніх
        UKF-фільтрів (їхні x та x_prior — основа sigma-точок наступного
        predict) та до mixed-стану imm (репортиться в overlay). Без цього
        контамінована vz екстраполює Z до 30-42 м (див. enable_coast_z_fence).
        P не чіпаємо — за нього відповідає Gate C (coast_cov_cap)."""
        zlo, zhi = self.coast_z_min, self.coast_z_max

        def _clamp(x):
            if x[6] > zhi:
                x[6] = zhi
                if x[7] > 0.0:
                    x[7] = 0.0
                    x[8] = 0.0
            elif x[6] < zlo:
                x[6] = zlo
                if x[7] < 0.0:
                    x[7] = 0.0
                    x[8] = 0.0

        for f in self.imm.filters:
            _clamp(f.x)
            _clamp(f.x_prior)
        _clamp(self.imm.x)
        _clamp(self.imm.x_prior)

    def predict(self):
        # 9D state: y = x[3], vy = x[4] (було 6D: x[2], x[3]).
        self.imm.M = get_dynamic_transition_matrix(
            self.imm.x[3], self.imm.x[4],
            mahalanobis_sq=self._last_mahalanobis_sq,
            frames_since_update=self.time_since_update,
            hit_residual_min_sq=self.hit_residual_min_sq,
            hit_residual_max_sq=self.hit_residual_max_sq,
            bounce_height_max=self.bounce_height_max,
            bounce_max_coast=self.bounce_max_coast,
            m_hit_target=self.m_hit_target,
        )
        self.imm.predict()
        if self._pending is not None:
            self._pending_frames_elapsed += 1
            if self._pending_frames_elapsed > self.HYSTERESIS_WINDOW:
                # Вікно підтвердження вичерпане — скидаємо pending і
                # відновлюємо нормальне старіння.
                self._pending = None
                self._pending_frames_elapsed = 0
                self.time_since_update += 1
            # else: трек не старіє під час активного вікна hysteresis
        else:
            self.time_since_update += 1

    def update(self, z, R=None):
        # R — опційна per-detection коваріація вимірювання (адаптивний σ_Z).
        # None → IMM-фільтри беруть власний фіксований self.R.
        current_mah_sq = self._last_mahalanobis_sq

        # ── Phase B: Delayed-accept hysteresis ────────────────────────────────
        # Defer-механізм (pending) увімкнено лише при enable_hysteresis.
        # Coast-skip (через _missed_streak) лишається активним завжди — це
        # обробка повернення з оклюзії, а не відкладання детекції.
        _coast_skip = False  # True → пропустити inflate та hysteresis-тригер

        if self.enable_hysteresis and self._pending is not None:
            pending_z, _ = self._pending
            self._pending = None
            self._pending_frames_elapsed = 0

            if current_mah_sq >= self.HYSTERESIS_CONFIRM_THRESHOLD:
                # Обидва кадри далеко від прогнозу → реальний маневр підтверджено.
                # Inflate P + update з поточним z.
                # pending_z (пізній вимір) не застосовуємо: подвійний imm.update()
                # без predict() між ними порушує PD-властивість P → Cholesky fail.
                # Inflate + поточний z достатньо для фіксації нової траєкторії.
                self._inflate_covariance()
                self.imm.update(z, R=R)
                self.time_since_update = 0
                self.hits += 1
                self.hit_streak += 1
                if self.state == 'Tentative' and self.hits >= self.min_hits:
                    self.state = 'Confirmed'
                self._missed_streak = 0
                return

            # mah_sq < CONFIRM_THRESHOLD → поточна детекція повернулась до
            # нормальної траєкторії → pending був false-positive.
            # Продовжуємо нормальним update без inflate та без нового тригера.
            _coast_skip = True
            self._missed_streak = 0

        elif self._missed_streak >= self.COAST_THRESHOLD:
            # Трек довго не бачив м'яча → висока mah_sq є ОЧІКУВАНОЮ
            # (коваріація P зросла природно під час коастингу), а не
            # ознакою маневру. Пропускаємо inflate і hysteresis-тригер.
            _coast_skip = True
            self._missed_streak = 0

        elif (self.enable_hysteresis and z is not None
              and current_mah_sq > self.HYSTERESIS_MAH_SQ_THRESHOLD):
            # Підозріла детекція — відкладаємо її на підтвердження.
            self._pending = (np.asarray(z, dtype=float).copy(), current_mah_sq)
            self._pending_frames_elapsed = 0
            self.imm.update(None)   # екстраполяція замість прийняття
            self.hit_streak = 0
            self._last_mahalanobis_sq = 0.0
            self._missed_streak += 1
            return
        # ── End Phase B ────────────────────────────────────────────────────────

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
                # 9D state: швидкісні індекси [1, 4, 7] (було 6D: [1, 3, 5]).
                # Accel-індекси (2, 5, 8) НЕ чіпаємо — лишаються 0 з
                # create_imm_estimator (немає підстав bootstrap'ити).
                for f in self.imm.filters:
                    f.x[1] = v_boot[0]
                    f.x[4] = v_boot[1]
                    f.x[7] = v_boot[2]
                    f.x_prior[1] = v_boot[0]
                    f.x_prior[4] = v_boot[1]
                    f.x_prior[7] = v_boot[2]
                # mixed state теж оновлюємо.
                self.imm.x[1] = v_boot[0]
                self.imm.x[4] = v_boot[1]
                self.imm.x[7] = v_boot[2]
                self.imm.x_prior[1] = v_boot[0]
                self.imm.x_prior[4] = v_boot[1]
                self.imm.x_prior[7] = v_boot[2]
                self._velocity_bootstrapped = True

        # Step 2.A — Track-level covariance reset on radical residual.
        # Виконується ПІСЛЯ Mechanism A bootstrap (щоб не конфліктувати:
        # у bootstrap'ному кадрі м'яч ще не "робив" удар, він просто
        # підхопив реальну швидкість), але ПЕРЕД imm.update(z).
        # _coast_skip = True означає, що висока mah_sq є очікуваною
        # (коастинг або resolved FP pending) — inflate не потрібен.
        do_cov_reset = (
            self.enable_cov_reset
            and z is not None
            and not _coast_skip
            and self._last_mahalanobis_sq > self.COV_RESET_MAH_SQ_THRESHOLD
        )
        if do_cov_reset:
            self._inflate_covariance()

        self.imm.update(z, R=R)
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        if self.state == 'Tentative' and self.hits >= self.min_hits:
            self.state = 'Confirmed'
        self._missed_streak = 0

    def mark_missed(self):
        self.hit_streak = 0
        # На кадрах без вимірювання residual не визначений — скидаємо у 0,
        # щоб у наступному predict() hit-тригер не активувався фантомно
        # з застарілих даних.
        self._last_mahalanobis_sq = 0.0
        self._missed_streak += 1  # Phase B: відстежуємо серію пропусків
        self.imm.update(None)     # Екстраполяція без вимірювання
        # Фікс кадрів 611-650: гасимо залишковий accel, щоб коастинг
        # не розганявся через інтеграцію застарілого ax у швидкість.
        self._collapse_accel_to_ballistic()
        # Gate C: кап росту P-діагоналі під час коастингу. Без нього P_vel
        # розбухає (2.5→700 за ~57 кадрів), сигма-точки UKF розкидаються
        # ±17-26 м/с, квадратичний drag (Jensen) тягне середню швидкість до
        # нуля й перебиває гравітацію → vy «замерзає», м'яч зависає замість
        # балістичної дуги. Кап тримає екстраполяцію фізичною за межами кадру.
        if self.enable_coast_cov_cap:
            self._cap_coast_covariance()
        # Coast Z-fence: тримаємо екстрапольовану глибину фізичною (≤ майданчик).
        if self.enable_coast_z_fence:
            self._fence_coast_depth()

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
    def __init__(self, dt=0.02, max_age=80, min_hits=3, gating_threshold=16.0,
                 bootstrap_max_gap_frames=5, bootstrap_max_dist=2.5,
                 bootstrap_max_speed=35.0, enable_hysteresis=False,
                 enable_cov_reset=False, enable_single_ball_nms=True,
                 enable_physical_gate=True, max_assoc_residual=3.0,
                 enable_coast_cov_cap=False, spawn_suppress_max_coast=0,
                 enable_depth_robust_gate=False, depth_hold_gain=0.5,
                 depth_innov_free=0.0,
                 enable_coast_z_fence=False, coast_z_min=-3.0, coast_z_max=21.0,
                 tentative_max_age=5, floor_kill_y=-0.3,
                 enable_adaptive_depth_R=False,
                 hit_residual_min_sq=8.0, hit_residual_max_sq=11.34,
                 bounce_height_max=0.55, bounce_max_coast=3,
                 m_hit_target=None):
        """
        :param dt:                    крок часу (= 1 / FPS).
        :param max_age:               к-сть кадрів без оновлення до
                                      видалення треку.
                                      Історія: 150 (sweep) → 15 (Plan A) →
                                      80 (Phase B + edge-gate, чиста
                                      розгортка).
                                      Старий 150 давав 3-секундний
                                      коастинг — підтверджений трек, що
                                      втратив детекцію, балістично летiв
                                      через увесь кадр i накопичував
                                      "фантомнi" пiдтвердженi треки на
                                      сміттєвих детекціях. Plan A зрізав до
                                      15, щоб їх убити. Але edge-gate
                                      (court-bounds у run_segment) тепер
                                      прибирає самі викиди в джерелі, тож
                                      головна причина для 15 зникла.
                                      Розгортка з гейтом: 80 дає найкращу
                                      fragmentation (0.235) + coverage
                                      (0.907) при mode-switch 2.54 Гц
                                      (нижче історичної бази ~3.5);
                                      повний ~13-сек трек відновлюється.
                                      100 доміновано 80-кою.
        :param min_hits:              к-сть кадрів для підтвердження
                                      треку (Tentative → Confirmed).
        :param gating_threshold:      χ²-поріг гейтингу Махаланобіса.
                                      χ²₃ p=0.99 = 11.34; default 16.0
                                      пускає прикордонні детекції маневру
                                      (mah 12-15) асоціюватись → оживляє
                                      hit-режим без зростання проліферації.
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
        :param enable_hysteresis:     вмикає Phase B delayed-accept
                                      hysteresis у кожному Track. Default
                                      False (див. Track.__init__ — defer
                                      канібалізує cov-reset+hit). Прокидається
                                      з run_segment через --enable_hysteresis.
        :param enable_cov_reset:      вмикає Step 2.A covariance reset у
                                      кожному Track. Default False (Path B),
                                      щоб оживити IMM hit-режим. run_segment
                                      --enable_cov_reset повертає назад.
        :param enable_single_ball_nms: Phase C. Домен — один м'яч, тож
                                      одночасно має бути лише ОДИН Confirmed-
                                      трек. Якщо їх >1 (фантом-дублі від
                                      сміттєвих спавнів / дивергентного
                                      коастингу), лишаємо найкращий, решту
                                      понижуємо до Tentative. Default True;
                                      run_segment --disable_single_ball_nms.
        :param enable_physical_gate:  Gate A. Коваріаційно-незалежний стелаж
                                      на евклідів residual ||z − прогноз||.
                                      Mahalanobis-гейт залежить від P, а P
                                      необмежено роздувається під час коастингу
                                      (tsu=47 → P_pos ~80-130) → стрибок-FP на
                                      8 м дає mah≈0.7 < гейт і ПРИЙМАЄТЬСЯ
                                      (катастрофа fr763/807/86: трек телепортує
                                      на сміттєву детекцію). Фіксований стелаж
                                      на residual незалежний від коастингу й
                                      ловить fr763 (resid 7.95) так само, як
                                      короткий коастинг ловив fr867 (mah=33).
                                      Default True; run_segment
                                      --disable_physical_gate.
        :param max_assoc_residual:    поріг Gate A (м). Розподіл прийнятих
                                      апдейтів: p99=7.95, але реальний-маневр
                                      максимум ≈2.2 м (fr283 resid 2.19,
                                      mah 13; подача fr650 resid 2.08); чиста
                                      прірва 2.2 → 5.95 м до найближчого
                                      телепорта. 3.0 м: +0.8 м над легіт-
                                      максимумом, −3 м під телепортом.
        """
        self.dt = dt
        self.max_age = max_age
        self.min_hits = min_hits
        self.gating_threshold = gating_threshold

        self.bootstrap_max_gap_frames = int(bootstrap_max_gap_frames)
        self.bootstrap_max_dist = float(bootstrap_max_dist)
        self.bootstrap_max_speed = float(bootstrap_max_speed)
        self.enable_hysteresis = bool(enable_hysteresis)
        self.enable_cov_reset = bool(enable_cov_reset)
        self.enable_single_ball_nms = bool(enable_single_ball_nms)
        # Gate A: коваріаційно-незалежний фізичний стелаж на residual.
        self.enable_physical_gate = bool(enable_physical_gate)
        self.max_assoc_residual = float(max_assoc_residual)
        # Gate C: кап росту P-діагоналі під час коастингу (передається у Track).
        self.enable_coast_cov_cap = bool(enable_coast_cov_cap)
        # Gate D: придушення спавну паразитного треку, поки реальний Confirmed-
        # трек КОРОТКО коастить. Домен — один м'яч. Коли м'яч на 1-2 кадри
        # зникає (руки гравця, злиплий/стиснутий bbox), його w колапсує →
        # implied Z стрибає на 5-11 м → Gate A (правильно) робить реальний трек
        # коастуючим. АЛЕ та сама стрибуча детекція раніше спавнила КОНКУРЕНТА,
        # який за min_hits кадрів ставав Confirmed і через single-ball NMS
        # (tsu_tol=1) ПЕРЕХОПЛЮВАВ ідентичність у 48-хітового реального треку —
        # filtZ телепортувало 6→16 м, ще й паразитна vz з bootstrap. Цей гейт
        # НЕ дає народитись конкуренту, поки існує живий Confirmed-трек із
        # 0 < tsu ≤ spawn_suppress_max_coast: викид просто ігнорується, а
        # коастуючий трек ре-захоплює м'яч, щойно повернеться валідна (Gate A)
        # детекція. Після цього порогу коасту м'яч вважається СПРАВДІ
        # переміщеним → спавн знову дозволено (легітимний handoff після довгої
        # втрати). 0 = вимкнено (A/B-сумісність). Рек. ~8 (вкриває типову
        # оклюзію руками ≤8 кадрів при 50 FPS).
        self.spawn_suppress_max_coast = int(spawn_suppress_max_coast)
        # Gate A-depth: depth-robust Gate A. Розкладаємо residual на ЛАТЕРАЛЬ
        # (X=idx0 ширина, Y=idx1 висота — це проєкція пікселя u,v, яку YOLO дає
        # коректно навіть при оклюзії) та ГЛИБИНУ (Z=idx2 — монокулярна
        # f·D/w, що деградує, коли w колапсує під час оклюзії руками).
        # При оклюзії residual латералі лишається <1 м, а residual глибини
        # стрибає на 4-11 м. Звичайний Gate A (||z−прогноз||) відкидає таку
        # детекцію цілком → реальний трек коастить, ВТРАЧАЮЧИ хороше u,v
        # (траєкторія відстає від м'яча — скарга користувача на Gate D).
        # depth-robust замість цього: (1) гейтить ЛИШЕ латераль жорстко
        # (hypot(dX,dY) ≤ max_assoc_residual) — детекцію з правильним u,v
        # ПРИЙМАЄМО; (2) велику інновацію глибини НЕ відкидаємо, а сильно
        # НЕДОВІРЯЄМО: роздуваємо σ_Z² на (depth_hold_gain·depth_innov)² і в
        # гейтингу (Mahalanobis), і на update-кроці. Глибину тоді тримає
        # прогноз (без телепорту 16 м), а латераль X,Y оновлюється з валідної
        # детекції (траєкторія приклеєна до м'яча, як у overlay_D027). Реальний
        # трек з'їдає детекцію → фантом не спавниться → Gate D непотрібен.
        # Default False (A/B-сумісність). depth_hold_gain≈0.5: при innov=8 м
        # σ_Z_eff≈4 м → детекція майже не зсуває depth, прогноз домінує.
        self.enable_depth_robust_gate = bool(enable_depth_robust_gate)
        self.depth_hold_gain = float(depth_hold_gain)
        # Мертва зона (Huber-style) для depth-distrust: демпфуємо лише
        # НАДЛИШОК |innov| понад цей поріг нормальної фізики. excess =
        # max(0, |innov| − depth_innov_free); σ_Z² += (gain·excess)².
        # МОТИВАЦІЯ: чисте (gain·innov)² душило і легітимну глибину —
        # на дальній стороні корту (rawZ природно шумить ±1.5-2 м →
        # filtZ залипав занизько) та на подачі (innov −4.5 м → filtZ повз
        # 12 кадрів замість швидкої збіжності). Мертва зона ~3 м пускає
        # нормальну фізику без демпфування, а давить лише викиди оклюзії
        # (innov 6-11 м). 0.0 = вимкнено (чисте квадратичне демпфування,
        # A/B-сумісність). Рек. ~3.0 (= max_assoc_residual: «нормальний»
        # фізичний крок).
        self.depth_innov_free = float(depth_innov_free)
        # Coast Z-fence (2026-06-04): під час коасту немає вимірювання → жоден
        # гейт не діє, а predict() екстраполює світову Z за контамінованою vz
        # (біля межі кадру w_box обвалюється → Z=f·D/w роздувається → vz±19 м/с).
        # На даних коаст ніс м'яч у Z=33-42 м (майданчик лише 18 м). Фенс у
        # mark_missed кожного треку клампить Z прогнозу в [coast_z_min,
        # coast_z_max] і гасить vz/az за фенсом. Default False (A/B-сумісність).
        # Рек. coast_z_max≈21 (= back line 18 м + ~3 м на сервера за лінією),
        # coast_z_min≈-3 (подача з-за ближньої лінії). Доповнює far-Z стелю
        # детекції (--court_z_far_max) у run_segment: одна давить джерело
        # (контамінований вимір), друга — екстраполяцію.
        self.enable_coast_z_fence = bool(enable_coast_z_fence)
        self.coast_z_min = float(coast_z_min)
        self.coast_z_max = float(coast_z_max)
        # Дефект 1: привид-треки. Tentative-трек, що не дійшов до min_hits,
        # не повинен коастити як повноцінний — 1-2 шумові детекції інакше
        # породжують ~80-кадрову фікцію (tid3/5/7: 0 confirmed, 99% coast,
        # під підлогою). Вбиваємо Tentative після цього к-ва коаст-кадрів.
        self.tentative_max_age = int(tentative_max_age)
        # Дефект 2: підпідлоговий коаст. М'яч у польоті не може бути нижче
        # підлоги; якщо трек НА КОАСТІ екстраполює y < floor_kill_y —
        # екстраполяція стала фікцією (tid2 пірнав до −5.6 м), термінуємо.
        # Невеликий від'ємний запас проти шуму біля підлоги (м'яч r≈0.1 м).
        self.floor_kill_y = float(floor_kill_y)
        # Phase F: адаптивний σ_Z. Якщо True й у update() передано
        # detection_covs — гейтинг і IMM-апдейт використовують per-detection
        # 3×3 R (анізотропний, витягнутий уздовж променя; σ_Zc = f·D/w²·σ_w),
        # а не фіксований self.R_matrix. Чесна довіра до глибини залежно від
        # ширини bbox. Default False (A/B-сумісність).
        self.enable_adaptive_depth_R = bool(enable_adaptive_depth_R)
        # IMM-тригери (Phase F експеримент): прокидаються у кожен Track при
        # створенні → get_dynamic_transition_matrix. Дозволяють тюнити
        # активацію bounce/hit-режимів з CLI без правок коду. Дефолти =
        # сигнатура get_dynamic_transition_matrix (A/B-сумісність).
        self.hit_residual_min_sq = float(hit_residual_min_sq)
        self.hit_residual_max_sq = float(hit_residual_max_sq)
        self.bounce_height_max = float(bounce_height_max)
        self.bounce_max_coast = int(bounce_max_coast)
        self.m_hit_target = (None if m_hit_target is None
                             else np.asarray(m_hit_target, dtype=float))
        # Gate B: merge-identity guard. Два Confirmed-треки успадковують
        # спільний id лише якщо фізично примиренні (той самий м'яч,
        # роздроблений маневром: ≤ цієї відстані). Далекий FP-трек, що
        # дожив до Confirmed, НЕ донорить id реальному коастуючому треку
        # (інакше merge-identity амплифікувала б телепорт). Маневр-фрагменти
        # ≤~2 м, FP-телепорти ≥6 м → 4.0 м чисто розділяє.
        self.SINGLE_BALL_MERGE_MAX_DIST = 4.0
        # Phase C: коастинг ≤ цієї к-сті кадрів не коштує ідентичності
        # (bucket'имо tsu). tol=1: реальний трек переживає ОДИН пропущений
        # кадр (детекційний шум), але після 2+ кадрів коастингу поступається
        # треку, що активно отримує детекції. Маленький tol важливий: під
        # час маневру програшний трек коастить по старій траєкторії й
        # ВІДХОДИТЬ від м'яча — чим раніше перемкнутись на детектований
        # трек, тим менший просторовий стрибок на switch'і.
        self.SINGLE_BALL_NMS_TSU_TOL = 1

        self.tracks = []
        self.next_id = 1
        # Матриця для гейтингу Махаланобіса. Узгоджуємо з R_init у
        # create_imm_estimator (σ_z ≈ 0.7 м для моно-камерної глибини).
        self.R_matrix = np.diag([0.02, 0.02, 0.5])

        # Ring-буфер невідповідних (unassigned) детекцій з кількох останніх
        # кадрів. Використовується для bootstrap'у швидкості при створенні
        # нового треку: якщо поточна незасоційована детекція близька до
        # детекції з останніх N кадрів, ініціалізуємо новий трек з
        # v = (z_now − z_prev) / (Δfr · dt) замість нулів.
        self._spawn_buffer = []  # елементи: (np.array z, int frame_idx)
        self._frame_counter = 0

    def update(self, detections_3d, detection_covs=None):
        """
        detections_3d: список масивів np.array([x, y, z]) для
        всіх знайдених об'єктів у кадрі
        detection_covs: опційний паралельний список 3×3 коваріацій
            вимірювання (по одній на детекцію). Вживається лише якщо
            enable_adaptive_depth_R=True. None → фіксований self.R_matrix.
        """
        self._frame_counter += 1

        def _R_for(d_idx):
            """R для d-ї детекції: адаптивна, якщо ввімкнено й надано."""
            if (self.enable_adaptive_depth_R
                    and detection_covs is not None
                    and detection_covs[d_idx] is not None):
                return detection_covs[d_idx]
            return self.R_matrix

        def _depth_robust_R(track, z, R_base):
            """Роздуваємо σ_Z² пропорційно квадрату інновації глибини, щоб
            великий стрибок глибини майже не зсував фільтр (трек тримає
            глибину прогнозом, латераль X,Y оновлює з валідної детекції).
            z_pred беремо з x_prior (predict уже викликано для всіх треків)."""
            z_pred = np.dot(track.h_matrix, track.imm.x_prior)
            depth_innov = float(z[2] - z_pred[2])
            # Мертва зона: душимо лише надлишок |innov| понад поріг
            # нормальної фізики, щоб не калічити легітимну глибину.
            excess = max(0.0, abs(depth_innov) - self.depth_innov_free)
            R_eff = np.array(R_base, dtype=float, copy=True)
            R_eff[2, 2] += (self.depth_hold_gain * excess) ** 2
            return R_eff

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
            # Gate A: прогнозована позиція треку (h·x_prior) — точка відліку
            # для коваріаційно-незалежного residual-стелажу.
            z_pred = np.dot(track.h_matrix, track.imm.x_prior)
            for d, z in enumerate(detections_3d):
                R_d = _R_for(d)
                # Gate A: відкидаємо пару, якщо евклідів стрибок від прогнозу
                # фізично неможливий за один крок асоціації — НЕЗАЛЕЖНО від
                # того, наскільки роздулась P під час коастингу (саме це
                # відкривало Mahalanobis-гейт для телепортів fr763/807/86).
                if self.enable_physical_gate:
                    if self.enable_depth_robust_gate:
                        # Gate A-depth: гейтимо ЛИШЕ латераль (X,Y). Велику
                        # інновацію глибини не відкидаємо — нею керує
                        # depth-distrust (роздутий σ_Z і в гейтингу, і на
                        # update). Так детекція з валідним u,v приймається.
                        lateral = float(np.hypot(z[0] - z_pred[0],
                                                 z[1] - z_pred[1]))
                        if lateral > self.max_assoc_residual:
                            continue
                        R_d = _depth_robust_R(track, z, R_d)
                    elif np.linalg.norm(z - z_pred) > self.max_assoc_residual:
                        continue
                dist_sq = track.get_mahalanobis_distance(z, R_d)
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
                R_c = _R_for(c)
                if self.enable_depth_robust_gate:
                    # Та сама роздута σ_Z, що й у гейтингу: депт тримається
                    # прогнозом, латераль X,Y оновлюється з детекції.
                    R_c = _depth_robust_R(self.tracks[r],
                                          detections_3d[c], R_c)
                self.tracks[r].update(detections_3d[c], R=R_c)
                unmatched_tracks.remove(r)
                unmatched_detections.remove(c)

        # 5. Обробка пропущених треків
        for t in unmatched_tracks:
            self.tracks[t].mark_missed()

        # 6. Створення нових треків для нерозпізнаних детекцій
        #    (із bootstrap'ом швидкості, якщо знайдено попередника).
        # Gate D: якщо реальний Confirmed-трек ЩОЙНО коастить коротко
        # (0 < tsu ≤ spawn_suppress_max_coast), не народжуємо конкурента з
        # викидних детекцій — нехай реальний трек ре-захопить м'яч. Викид НЕ
        # потрапляє і в spawn-буфер → пізніший легітимний спавн стартує з
        # нульовою швидкістю (без паразитного vz-bootstrap).
        suppress_spawn = (
            self.spawn_suppress_max_coast > 0
            and any(
                t.state == 'Confirmed'
                and 0 < t.time_since_update <= self.spawn_suppress_max_coast
                for t in self.tracks
            )
        )
        if not suppress_spawn:
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
                      min_hits=self.min_hits,
                      enable_hysteresis=self.enable_hysteresis,
                      enable_cov_reset=self.enable_cov_reset,
                      enable_coast_cov_cap=self.enable_coast_cov_cap,
                      enable_coast_z_fence=self.enable_coast_z_fence,
                      coast_z_min=self.coast_z_min,
                      coast_z_max=self.coast_z_max,
                      hit_residual_min_sq=self.hit_residual_min_sq,
                      hit_residual_max_sq=self.hit_residual_max_sq,
                      bounce_height_max=self.bounce_height_max,
                      bounce_max_coast=self.bounce_max_coast,
                      m_hit_target=self.m_hit_target)
            )
            self.next_id += 1
            # Видаляємо використану spawn-точку.
            del self._spawn_buffer[best_idx]
        else:
            # Класичний spawn з нульовою швидкістю + кладемо детекцію
            # у буфер для майбутнього bootstrap'у.
            self.tracks.append(
                Track(self.next_id, z_now_arr, self.dt,
                      min_hits=self.min_hits,
                      enable_hysteresis=self.enable_hysteresis,
                      enable_cov_reset=self.enable_cov_reset,
                      enable_coast_cov_cap=self.enable_coast_cov_cap,
                      enable_coast_z_fence=self.enable_coast_z_fence,
                      coast_z_min=self.coast_z_min,
                      coast_z_max=self.coast_z_max,
                      hit_residual_min_sq=self.hit_residual_min_sq,
                      hit_residual_max_sq=self.hit_residual_max_sq,
                      bounce_height_max=self.bounce_height_max,
                      bounce_max_coast=self.bounce_max_coast,
                      m_hit_target=self.m_hit_target)
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

            # Дефект 1: привид-трек. Tentative (так і не підтвердився), а вже
            # коастить понад tentative_max_age кадрів → шумовий спавн, який без
            # цього малював би 80-кадрову фікцію. Підтверджені треки не зачеплені.
            elif (track.state == 'Tentative'
                  and track.time_since_update > self.tentative_max_age):
                track.state = 'Deleted'

            # Дефект 2: підпідлоговий коаст. Лише коли трек БЕЗ виміру (коаст):
            # за наявності детекції y вимірюється й клампінг не потрібен.
            elif (track.time_since_update > 0
                  and track.imm.x[3] < self.floor_kill_y):
                track.state = 'Deleted'

            # Трек залишається, якщо він не видалений
            if track.state != 'Deleted':
                active_tracks.append(track)

        self.tracks = active_tracks

        if self.enable_single_ball_nms:
            self._suppress_duplicate_confirmed()

    def _suppress_duplicate_confirmed(self):
        """Phase C: single-ball NMS. Домен — один м'яч у грі, тож
        одночасно має бути лише ОДИН Confirmed-трек. Якщо їх кілька
        (фантом-дублі від сміттєвих спавнів або дивергентного коастингу),
        лишаємо найкращий, решту ВИДАЛЯЄМО.

        Видалення (а не пониження до Tentative) — навмисне: понижений
        дубль продовжує красти детекції в Угорському алгоритмі (на маневрі
        детекція перескакує між двома треками, що тримають той самий м'яч),
        re-confirm'иться і знову конкурує — дубль жив ~200 кадрів і давав
        9-14 стрибків ідентичності. Видалення зупиняє це: детекція стабільно
        йде на переможця; якщо ж переможець справді розходиться з м'ячем,
        нова детекція спавнить свіжий трек (ціна — трохи вища fragmentation,
        але траєкторія помітно гладша: jerk↓, switches 9→3).

        Найкращий = (min bucketed-tsu, max hits, min останній
        Mahalanobis^2): активно оновлюваний > довгоживучий > добре
        вписаний.

        ВАЖЛИВО — bucketing tsu: короткий коастинг (tsu ≤ TSU_TOL)
        прирівнюється до 0. Без цього один пропущений кадр на реальному
        треку (tsu=1) віддавав би ідентичність дивергентному привиду з
        tsu=0 — навіть якщо привид має 4 hits проти 245 у реального.
        Це спричиняло мерехтіння ідентичності під час маневру (детекції
        на 2-3 кадри перескакували на привид-трек). З bucket'ом реальний
        трек (на порядок більше hits) тримає ідентичність крізь коротку
        турбулентність; привид із ДОВГИМ коастингом (tsu > TSU_TOL) усе ж
        поступається активному треку — коректний handoff."""
        confirmed = [t for t in self.tracks if t.state == 'Confirmed']
        if len(confirmed) < 2:
            return

        def _key(t):
            eff_tsu = 0 if t.time_since_update <= self.SINGLE_BALL_NMS_TSU_TOL \
                else t.time_since_update
            return (eff_tsu, -t.hits, t._last_mahalanobis_sq)

        confirmed.sort(key=_key)
        winner = confirmed[0]

        # Merge-identity (фікс fragmentation): домен — один м'яч, тож усі
        # співіснуючі Confirmed-треки = той САМИЙ фізичний м'яч, лише
        # роздроблений маневром (детекція вистрибує з гейту → народжується
        # конкурент). Переможець успадковує МІНІМАЛЬНИЙ id групи →
        # логічна ідентичність неперервна через шов, а не новий id щоразу.
        # Розділені реальним розривом траєкторії (різні розіграші) НІКОЛИ
        # не співіснують як Confirmed (gap десятки кадрів) → сюди не
        # потрапляють → хибного злиття немає. Переносимо й МАКС hits, щоб
        # злитий трек лишався "досвідченим" і стабільним переможцем у
        # наступних NMS (tiebreaker -hits).
        #
        # Gate B (merge guard): id успадковуємо ЛИШЕ від фізично примиренних
        # членів групи (≤ SINGLE_BALL_MERGE_MAX_DIST від переможця). Далекий
        # FP-трек, що дожив до Confirmed (пройшов min_hits-верифікацію), усе
        # одно НЕ донорить свій id — інакше merge-identity легалізувала б
        # телепорт (амплифікація проблеми fr763). Він просто видаляється
        # як дубль. Маневр-фрагменти лишаються близько (≤~2 м) → зливаються
        # нормально; реальний м'яч, що далеко "перестрибнув", дає новий id —
        # коректно, бо такий стрибок = фактично новий сегмент гри.
        def _pos(t):
            x = t.imm.x_post
            return np.array([x[0], x[3], x[6]])

        wp = _pos(winner)
        reconcilable = [
            t for t in confirmed
            if np.linalg.norm(_pos(t) - wp) <= self.SINGLE_BALL_MERGE_MAX_DIST
        ]
        winner.track_id = min(t.track_id for t in reconcilable)
        winner.hits = max(t.hits for t in reconcilable)

        for loser in confirmed[1:]:
            loser.state = 'Deleted'
        self.tracks = [t for t in self.tracks if t.state != 'Deleted']

    def get_confirmed_tracks(self):
        """Повертає позиції лише тих треків, в яких ми впевнені"""
        return [t for t in self.tracks if t.state == 'Confirmed']