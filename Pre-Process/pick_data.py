# pylint: disable = line-too-long, too-many-lines, too-many-arguments, wrong-import-order, invalid-name, missing-docstring

import csv

lookingLandmarkNames = []

with open("landmark_names_turkey.txt") as fi:
    for landmarkName in fi:
        lookingLandmarkNames.append(landmarkName.strip().upper())

with open("landmark_names_worldwide.txt") as fi:
    for landmarkName in fi:
        lookingLandmarkNames.append(landmarkName.strip().upper())

pickedLandmarks = []

with open('../Data/descriptions.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')

    for row in spamreader:
        if any(landmarkName in row[1].upper() for landmarkName in lookingLandmarkNames):
            pickedLandmarks.append(row)

print(len(pickedLandmarks), "landmarks added.")

with open('../Data/picked_landmarks.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    for row in pickedLandmarks:
        spamwriter.writerow(row)
