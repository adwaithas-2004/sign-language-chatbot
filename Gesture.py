import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open the webcam")

while True:
    ret, img = cap.read()
    if not ret:
        break
    cv2.imshow("My Video", img)

    # Press "Esc" (ASCII 27) to exit
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
