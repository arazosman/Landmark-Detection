import csv

mp = {}

with open('../Data/picked_landmarks.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')

    for row in spamreader:
        mp.update({row[0]: 0})

with open('../Data/train.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')

    for row in spamreader:
        if row[2] in mp:
            mp[row[2]] += 1

print(len(mp))

for m in mp:
    print(m, mp[m])