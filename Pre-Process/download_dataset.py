import csv
import cv2
import os
import urllib.request

def createFolders(landmarks):
    for id in landmarks:
        os.mkdir(os.path.join("../Data/train", id))

def downloadImage(path, index, order):
    img = cv2.imread(path)
    img = cv2.resize(img, (96, 96))
    cv2.imwrite(os.path.join("../Data/train", index, str(order) + ".jpg"), img)

landmarks = {}

with open('../Data/picked_landmarks.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')

    for row in spamreader:
        landmarks.update({row[0]: 0})

createFolders(landmarks)
'''
with open('../Data/train.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    i = 0
    problem = 0

    for row in spamreader:
        if row[2] in landmarks:
            try:
                i += 1
                print(i, row[2], row[1])
                urllib.request.urlretrieve(row[1], "curr.jpg")
                landmarks[row[2]] += 1
                downloadImage("curr.jpg", row[2], landmarks[row[2]]) 
            except:
                print("cannot download an image for index ", row[0])
                problem += 1

print("could not download", problem, "images")
'''