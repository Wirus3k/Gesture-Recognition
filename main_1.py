# Import bibliotek

import cv2
import numpy as np
from sklearn.metrics import pairwise

# Zdefiniowanie zmiennej przechowującej liczbę klatek do utworzenia zmiennej tlo oraz liczby klatek wymaganej do kalibracji
liczba_klatek = 0
liczba_klatek_kalibracja = 90

# Zdefiniowanie zmiennej Tlo oraz wagi wykorzystanej w proces
Tlo = None
waga = 0.5

# Ustawienie kwadratu ROI(Region of Interest) w
Top = 20
Bottom = 300
Right = 300
Left = 600

# Parametry blurowania Gaussowskiego

Gaussian_Par = 7

thresh_prog = 25

# Inicjalizacja kamery
cam = cv2.VideoCapture(0)

# Zdefiniowanie funkcji tworzącej uśrednione tło
def accum_weighted(frame, waga):
    global Tlo
    # If, która kopiuje pierwszą klatkę jako tło
    if Tlo is None:
        Tlo = frame.copy().astype("float")
        return None
    # Funkcja zliczająca średnią klatek na wejściu, accumulateWeighted(InputArray, InputOutputArray, alpha - waga input image)
    cv2.accumulateWeighted(frame, Tlo, waga)


def segment(frame):
    global Tlo

    # Obliczanie absolutnej roznicy pomiedzy Tlo a obecną klatką
    diff = cv2.absdiff(Tlo.astype("uint8"), frame)

    # Progowanie
    _, thresholded = cv2.threshold(diff, thresh_prog, 255, cv2.THRESH_BINARY)
    # Wyrysowanie konturu dloni na podstawie zbinaryzowanego obrazu
    contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Jeśli długość konturu jest zero wtedy nic nie zwracamy
    if len(contours) == 0:
        return None
    else:
        hand_segment = max(contours, key=cv2.contourArea)
        return (thresholded, hand_segment)

# Funkcja zliczania palców
def count_fingers(thresholded, hand_segment):
    # Zmienna przechowywująca 4  najbardziej wysunięte punkty
    max_points = cv2.convexHull(hand_segment)
    # Otoczka wypukła to zewnętrzny kontur, który zawiera wszystkie punkty zestawu punktów wewnątrz niego.
    # Otoczka jest zawsze wypukła, co oznacza, że żaden z jej kątów nie jest większy niż 180 stopni.


    Max_up = tuple(max_points[max_points[:, :, 1].argmin()][0])
    Max_bottom = tuple(max_points[max_points[:, :, 1].argmax()][0])
    Max_left = tuple(max_points[max_points[:, :, 0].argmin()][0])
    Max_right = tuple(max_points[max_points[:, :, 0].argmax()][0])

    # Znalezienie srodka dloni na podstawie max_cnt
    cX = (Max_left[0] + Max_right[0]) // 2
    cY = (Max_up[1] + Max_bottom[1]) // 2

    # Funkcją biblioteki scikit-learn i jest używana do obliczania odległości euklidesowych między punktami.
    # W naszym przypadku pomiędzy środkiem a punktami max
    odleglosc = pairwise.euclidean_distances([(cX, cY)], Y=[Max_left, Max_right, Max_up, Max_bottom])[0]
    max_distance = odleglosc.max()

    # Parametry okregu do nowej tablicy
    promien = int(0.85 * max_distance)
    obwod = (2 * np.pi * promien)

    # Linijka tworzy nową tablicę o wymiarach takich samych jak thresholded
    Kolo_ROI = np.zeros(thresholded.shape[:2], dtype="uint8")

    # Rysujemy okrąg na Kolo_ROI
    cv2.circle(Kolo_ROI, (cX, cY), promien, 255, 10)
    # Funkcja cv2.bitwise_and() jest używana do zastosowanie maski na obrazie z progowaniem, w celu odfiltrowania wszystkich elementów poza Dlonia.
    Kolo_ROI = cv2.bitwise_and(thresholded, thresholded, mask=Kolo_ROI)

    # Wyrysowanie konturu dloni na podstawie zbinaryzowanego obrazu
    contours, hierarchy = cv2.findContours(Kolo_ROI.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cv2.imshow("Kolo Roi", Kolo_ROI)
    cv2.drawContours(thresholded, contours, -1, (0, 255, 0), 2, 1)

    liczba_palcow = 0

    for punkty in contours:

        # Kwadratowy boks konturu
        (x, y, w, h) = cv2.boundingRect(punkty)

        # Dwa załozenia do zliczania palcy

        # 1. Obszar konturu nie jest dolną częścią dłoni (nadgarstek)
        nadgarstek = ((cY + (cY * 0.25)) > (y + h))

        # 2. Ilość punktów wzdłuż konturu nie przekracza 25% obwodu okrągłego ROI (inaczej punkty odliczamy z ręki)
        limit_points = ((obwod * 0.2) > punkty.shape[0])

        if nadgarstek and limit_points:
            #Inkrementacja liczby palcy
            liczba_palcow += 1

    return liczba_palcow




# Pętla odpowiedzialna za działanie programu
while True:
    #Przechwycenie klatki z kamery do zmiennej frame, a następnie wykonanie lustrzenego odbicia
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1) #Jedynka to odbicie pionowe
    Kwadrat_ROI = frame[Top:Bottom, Right:Left]
    cv2.putText(frame, 'Prog ' + str(thresh_prog), (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, 1)
    cv2.putText(frame, 'Gauss ' + str(Gaussian_Par), (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, 1)
    cv2.putText(frame, 'k - Kalibracja ', (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, 1)
    cv2.putText(frame, 'x - Prog +5 ', (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, 1)
    cv2.putText(frame, 'z - Prog -5 ', (15, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, 1)
    cv2.putText(frame, 'q - Wyjscie ', (15, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, 1)
    # Zastosowanie Skali szarości wraz z blurem gaussowskim
    gray = cv2.cvtColor(Kwadrat_ROI, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (Gaussian_Par, Gaussian_Par), 0)

    # Wywołanie funkcji accum_weighted dla pierwszy n (liczba_klatek_kalibracja) klatek
    if liczba_klatek <= liczba_klatek_kalibracja:
        accum_weighted(blur, waga)
        cv2.putText(frame,'Trwa kalibracja tla', (298, 350), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)

    else:
        hand = segment(blur)
        if hand is not None:
            thresholded, hand_segment = hand
            cv2.drawContours(frame, [hand_segment + (Right, Top)], -1, (0, 255, 0), 2, 1)
            fingers = count_fingers(thresholded, hand_segment)
            cv2.putText(frame, 'Ilosc wykrytych palcow ', (260, 350), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2, 1)
            cv2.putText(frame, str(fingers), (410, 440), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 0, 0), 3, 3)
            cv2.imshow("Thesholded", thresholded)
        else:
            cv2.putText(frame, 'Umiesc dlon w czerwonym kwadracie !', (15, 350), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0), 2, 1)



    # Utworzenie kwadratu ROI do zczytywania dłoni
    cv2.rectangle(frame, (Left, Top), (Right, Bottom), (0, 0, 255), 5)

    cv2.imshow("Program zliczajacy ilosc palcow w czasie rzeczywistym", frame)

    # Inkrementacja zmiennej, która przechowuje liczbę klatek
    liczba_klatek += 1

    if cv2.waitKey(1) == ord('q'):
        break

    if cv2.waitKey(1) == ord('k'):
        liczba_klatek = 0

    if cv2.waitKey(1) == ord('x'):
        if thresh_prog > 5 and thresh_prog < 123:
            thresh_prog += 5

    if cv2.waitKey(1) == ord('z'):
        if thresh_prog > 11 and thresh_prog < 127:
            thresh_prog -= 5


# Zamknięcie okien
cam.release()
cv2.destroyAllWindows()