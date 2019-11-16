import csv

st = set()

with open('../Data/picked_landmarks.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')

    for row in spamreader:
        st.add(row[0])

count = 0

with open('../Data/train.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')

    for row in spamreader:
        if row[2] in st:
            count += 1

print(count)