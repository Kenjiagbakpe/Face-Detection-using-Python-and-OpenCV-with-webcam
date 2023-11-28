import cv2
import numpy as np
import os

# Path for face image database
path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to get the images and label data
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []
    for imagePath in imagePaths:
        img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img)
        for (x, y, w, h) in faces:
            faceSamples.append(img[y:y+h, x:x+w])
            ids.append(id)
    return faceSamples, ids

print("\n[INFO] Training faces. It will take a few seconds. Wait ...")
faces, ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml
recognizer.save('trainer/trainer.yml')

# Print the number of faces trained and end the program
print("\n[INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
