import csv
import cv2
import os
import urllib.request

landmarks = set()

with open('../Data/picked_landmarks.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')

    for row in spamreader:
        landmarks.add(row[0])

trainCSV = open('../Data/train.csv', newline='')
spamreader = csv.reader(trainCSV, delimiter=',', quotechar='|')

newCSV = open('../Data/newCSV.csv', 'w', newline='')
spamwriter = csv.writer(newCSV, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

for row in spamreader:
    if row[2] in landmarks:
        spamwriter.writerow(row)
