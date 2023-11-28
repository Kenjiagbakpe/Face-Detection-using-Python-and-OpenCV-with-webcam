import cv2
import os

# Create a video capture object
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # Set video width
cam.set(4, 480)  # Set video height

# Load the face detector cascade classifier
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Input user id as an integer
face_id = int(input('\nEnter user id and press <return> ==>  '))
print("\n[INFO] Initializing face capture. Look at the camera and wait ...")

# Initialize individual sampling face count
count = 0

while True:
    ret, img = cam.read()
    img = cv2.flip(img, 1)  # Flip video image horizontally (optional)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1
        # Save the captured image into the dataset folder
        cv2.imwrite(f"dataset/User.{face_id}.{count}.jpg", gray[y:y + h, x:x + w])
        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 30:  # Capture 30 face samples and stop video
        break

# Clean up
print("\n[INFO] Exiting program and cleaning up stuff")
cam.release()
cv2.destroyAllWindows()
