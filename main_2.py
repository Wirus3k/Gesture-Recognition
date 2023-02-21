import cv2 as cv
import numpy as np
import math
#image1 = cv.imread("zdjecie.jpg")
cap = cv.VideoCapture(0)
while cap.isOpened():
    ret, img = cap.read()
    cv.rectangle(img, (50, 75), (275, 300), (0, 0, 255), 2)  # Utworzenie kwadratu to zczytywania dłoni
    crop = img[75:300, 50:275]  # Skopiowanie pola kwadratu do procesu przetwarzania danych
    #difference = cv.subtract(crop, image1)
    grey = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)  # Zastosowanie Skali szarości
    gauss = cv.GaussianBlur(grey, (35, 35), 0)  # Blurowanie Gausowskie
    ret, thresh = cv.threshold(gauss, 127, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)  # Thresholding

    # Wyrysowanie konturu dloni na podstawie zbinaryzowanego obrazu
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # Maksymalne punkty konturu
    max_cnt = max(contours, key=cv.contourArea)
    cv.drawContours(crop, max_cnt, -1, (255, 0, 0), 2)

    # Na podstawie max_cnt wyrysowanie boksa
    convhull = cv.convexHull(max_cnt, returnPoints=True)
    cv.drawContours(crop, [convhull], -1, (0, 255, 0), 2, 1)

    # Znalezienie srodka dloni na podstawie max_cnt
    M = cv.moments(max_cnt)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # Wyrysowanie zoltego srodka dloni
        cv.circle(crop, (cX, cY), 6, (0, 255, 255), 1)

    # Aproksymacja konturu
    contour_poly = cv.approxPolyDP(max_cnt, 0.01 * cv.arcLength(max_cnt, True), True)
    # Wypuklosci
    hull = cv.convexHull(contour_poly, returnPoints=False)
    # Znalezienie punktow Wkleslych
    defects = cv.convexityDefects(contour_poly, hull)

    # Utworzenie petli zliczajacej i wyswietlajacej ilosc palcy
    if defects is not None:
        count_defects = 0;
        points = []
        for i in range(defects.shape[0]):
            start_index, end_index, far_pt_index, fix_dept = defects[i][0]
            start_pts = tuple(contour_poly[start_index][0])  # Rozpoczecie wkleslosci
            end_pts = tuple(contour_poly[end_index][0])  # Zakonczenie wkleslosci
            mid_pts = (int((start_pts[0] + end_pts[0]) / 2),
                       int((start_pts[1] + end_pts[1]) / 2))  # Srodek linni laczacej star i end
            far_pts = tuple(contour_poly[far_pt_index][0])  # punkt wkleslosci

            points.append(mid_pts)

            cv.circle(crop, mid_pts, 2, (204, 204, 255), 2)  # Rozowy
            cv.circle(crop, start_pts, 2, (0, 128, 0), 2)  # Zielony (BGR)
            cv.circle(crop, end_pts, 2, (0, 0, 255), 2)  # Czerwony
            cv.circle(crop, far_pts, 2, (0, 0, 0), 2)  # Czarny

            #cv.line(crop, start_pts, end_pts, (204, 0, 102), 2) #Fioletowa
            #cv.line(crop, start_pts, far_pts, (255, 255, 0), 2)#Jasny niebieski
            #cv.line(crop, end_pts, far_pts, (255,255,0), 2)#Jasny niebieski

            # Znajdujemy dlugosci kazdego z bokow
            a = np.sqrt((end_pts[0] - start_pts[0]) ** 2 + (end_pts[1] - start_pts[1]) ** 2)
            b = math.sqrt((far_pts[0] - start_pts[0]) ** 2 + (far_pts[1] - start_pts[1]) ** 2)
            c = math.sqrt((end_pts[0] - far_pts[0]) ** 2 + (end_pts[1] - far_pts[1]) ** 2)

            # Znajdujemy kat na podstawie twierdzenia cosinusow
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))

            # Wzor herona na pole trojkata
            s = (a + b + c) / 2
            ar = math.sqrt(s * (s - a) * (s - b) * (s - c))
            d = ar / a

            if angle <= 90 and (d > 5 and b > 5):
                count_defects += 1

        if count_defects == 1:
            cv.putText(img, "1 Palec", (100, 50), cv.FONT_HERSHEY_DUPLEX, 1, 3)
        elif count_defects == 2:
            cv.putText(img, "2 Palce", (100, 50), cv.FONT_HERSHEY_DUPLEX, 1, 3)
        elif count_defects == 3:
            cv.putText(img, "3 Palce", (100, 50), cv.FONT_HERSHEY_DUPLEX, 1, 3)
        elif count_defects == 4:
            cv.putText(img, "4 Palce", (100, 50), cv.FONT_HERSHEY_DUPLEX, 1, 3)
        elif count_defects == 5:
            cv.putText(img, "5 Palcy", (100, 50), cv.FONT_HERSHEY_DUPLEX, 1, 3)
        elif count_defects > 5:
            cv.putText(img, "5 Palcy", (100, 50), cv.FONT_HERSHEY_DUPLEX, 1, 3)
        else:
            cv.putText(img, "0 Palcy", (100, 50), cv.FONT_HERSHEY_DUPLEX, 1, 3)

    cv.putText(img, "Dopasuj dlon do czerwonego kwadratu", (5, 450), cv.FONT_HERSHEY_DUPLEX, 1, (125, 246, 55), 1)
    cv.imshow('Gesture', img)
    cv.imshow('Crop', crop)
    cv.imshow('Grey', grey)
    cv.imshow('GaussianBlur', gauss)
    cv.imshow('Thresholding', thresh)

    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()