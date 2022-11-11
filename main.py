import cv2 as cv

cap = cv.VideoCapture(0)
while cap.isOpened():
    ret, img = cap.read()

    cv.rectangle(img, (50, 75), (275, 300), (0, 0, 255), 2)
    crop = img[75:300, 50:275] # Cropping to take image process
    grey = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
    gauss = cv.GaussianBlur(grey, (35, 35), 0)
    ret, thresh = cv.threshold(gauss, 127, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    cv.imshow('Gesture', img)
    cv.imshow('Crop', crop)
    cv.imshow('Grey', grey)
    cv.imshow('GaussianBlur', gauss)
    cv.imshow('Thresholding', thresh)


    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()