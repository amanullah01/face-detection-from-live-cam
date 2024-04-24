import cv2

# import trained module

# cascade_classifier = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")
cascade_classifier = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")

# capture video using webcam
video_capture = cv2.VideoCapture(0)

# setting width and height
video_capture.set(3, 640)
video_capture.set(4, 480)

while True:
    # returns next video frame
    ret, frame = video_capture.read()

    # transform to gray image
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = cascade_classifier.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=10, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 5)

    cv2.imshow("Face detect from live camera", frame)

    # exit
    key = cv2.waitKey(30) & 0xff
    if key == 27:
        break

video_capture.release()
cv2.destroyAllWindows()
