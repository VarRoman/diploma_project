ZRZUTY EKRANU DO SPRAWOZDANIA RoadVision
=========================================

Umieść w tym katalogu (Latex_docs/img/) 4 pliki PNG o poniższych nazwach.
Plik main.tex sam je podstawi w miejsca obecnych szarych placeholderów
(makro \zrzut). Aby pokazać prawdziwy obraz, w main.tex zastąp wywołanie
\zrzut{nazwa.png}{wysokosc}{opis} poleceniem:
    \includegraphics[width=0.8\textwidth]{img/nazwa.png}

Wymagane pliki i co powinny przedstawiać:

1) 01_widok_ogolny.png  (rys. 14)
   Cały ekran aplikacji MapView: ikona pojazdu, niebieska linia trasy oraz
   kilka markerów defektów rozsianych po mapie Rzeszowa.
   Zalecane: pełne okno aplikacji, szer. ok. 1200 px.

2) 02_markery.png  (rys. 15, lewa)
   Zbliżenie na 2-3 markery: dziura (czerwony), wyboj (pomaranczowy),
   gwaltowne hamowanie (szary). Widoczne ikony na tle ulicy.

3) 03_pogoda.png  (rys. 15, prawa)
   Widget pogodowy w lewym gornym rogu (piktogram + temperatura + wilgotnosc).
   Najlepiej kadr pokazujacy sam widget i fragment mapy.

4) 04_tryb_sledzenia.png  (rys. 16)
   Aplikacja w trybie sledzenia (przycisk "Sledzenie: WL." w prawym gornym
   rogu), mapa wycentrowana na pojezdzie.

Wskazowki techniczne:
- Format: PNG (ewentualnie JPG, wtedy zmien rozszerzenie w main.tex).
- Proporcje pozioma (landscape) dla rys. 14 i 16, dowolne dla rys. 15.
- Jak zrobic: uruchom caly system (docker compose up) + aplikacje MapView
  (python MapView/new_main.py), poczekaj az pojazd przejedzie kawalek trasy,
  zrob zrzut ekranu (np. PrintScreen / narzedzie do wycinania).
