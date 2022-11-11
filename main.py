import cv2 as cv

cap = cv.VideoCapture(0)
while cap.isOpened():
    ret, img = cap.read()
    cv.imshow('Gesture', img)
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()