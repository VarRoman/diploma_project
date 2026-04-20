import cv2
import numpy as np


# Порожня функція-заглушка для повзунків
def nothing(x):
    pass


def main():
    video_path = "data/test_match.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Помилка: Не вдалося відкрити відеофайл.")
        return

    # Створюємо вікно для повзунків
    cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Trackbars", 400, 300)

    # Ініціалізація повзунків: (Назва, Вікно, Стартове значення, Максимум, Функція)
    cv2.createTrackbar("H_MIN", "Trackbars", 0, 179, nothing)
    cv2.createTrackbar("S_MIN", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("V_MIN", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("H_MAX", "Trackbars", 179, 179, nothing)
    cv2.createTrackbar("S_MAX", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("V_MAX", "Trackbars", 255, 255, nothing)

    print("=== Інструкція ===")
    print("Пробіл - пауза/продовження відео.")
    print("Клавіша 'q' - вихід.")
    print("На паузі крути повзунки, щоб виділити м'яч білим кольором.")

    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                # Перезапуск відео по колу для зручності
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Зменшуємо кадр, щоб влазив на екран
            frame = cv2.resize(frame, (1024, 576))
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Зчитуємо поточні значення повзунків
        h_min = cv2.getTrackbarPos("H_MIN", "Trackbars")
        s_min = cv2.getTrackbarPos("S_MIN", "Trackbars")
        v_min = cv2.getTrackbarPos("V_MIN", "Trackbars")
        h_max = cv2.getTrackbarPos("H_MAX", "Trackbars")
        s_max = cv2.getTrackbarPos("S_MAX", "Trackbars")
        v_max = cv2.getTrackbarPos("V_MAX", "Trackbars")

        # Формуємо масиви для маски
        lower_bound = np.array([h_min, s_min, v_min])
        upper_bound = np.array([h_max, s_max, v_max])

        # Застосовуємо маску
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # Накладаємо маску на оригінальний кадр для наочності (показує кольори тільки там, де маска біла)
        result = cv2.bitwise_and(frame, frame, mask=mask)

        # Вивід вікон
        cv2.imshow("Original", frame)
        cv2.imshow("Mask", mask)
        cv2.imshow("Result", result)

        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            # Перед виходом друкуємо знайдені значення в термінал
            print("\n=== ЗБЕРЕЖИ ЦІ ЗНАЧЕННЯ ДЛЯ ОСНОВНОГО КОДУ ===")
            print(f"lower_color = np.array([{h_min}, {s_min}, {v_min}])")
            print(f"upper_color = np.array([{h_max}, {s_max}, {v_max}])")
            break
        elif key == ord(' '):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()