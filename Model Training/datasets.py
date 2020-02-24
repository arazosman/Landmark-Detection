# pylint: disable = line-too-long, too-many-lines, too-many-arguments, wrong-import-order, invalid-name, missing-docstring, bare-except

import os
import numpy as np
from PIL import Image
from skimage import color
import h5py

def getClasses(path):
    classes = {}
    index = 0

    for className in os.listdir(path):
        classes[className] = index
        index += 1

    return classes

###################################

def shuffleDataset(X, Y, rate):
    m = X.shape[0]
    #np.random.seed(0)
    perm = np.random.permutation(m)
    print(X.shape)
    #print(X[100][5][4])

    X_shuffled = X[perm, :, :, :]
    Y_shuffled = Y[:, perm]

    m_train = int(m*rate)

    X_train = X_shuffled[:m_train, :, :, :]
    Y_train = Y_shuffled[:, :m_train]
    X_test = X_shuffled[m_train:, :, :, :]
    Y_test = Y_shuffled[:, m_train:]

    return (X_train, Y_train), (X_test, Y_test)

###################################

def getDataset(path, classes, pixel=32, rate=0.8):
    """
    A function which reads images and resize them.
    - pixel: width and height of the images after resizing
    """
    X = []
    Y = []

    # getting images:
    for root, _, files in os.walk(path):
        for file in files:
            imagePath = os.path.join(root, file)
            className = os.path.basename(root)

            try:
                image = Image.open(imagePath)
                image = np.asarray(image)
                image = np.array(Image.fromarray(image.astype('uint8')).resize((pixel, pixel)))

                if len(image.shape) == 2 or image.shape[2] == 1: # grayscale
                    image = color.gray2rgb(image)
                elif image.shape[2] == 4:                        # alpha
                    image = color.rgba2rgb(image)

                X.append(image)
                Y.append(classes[className])
            except:
                print(file, "could not be opened")

    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.int16).reshape(1, -1)

    return shuffleDataset(X, Y, rate)

###################################

def saveDataset(X_train, Y_train, X_test, Y_test, path):
    """
    A function which saves numpy arrays of a dataset to binary files.
    """
    np.save(os.path.join(path, "X_train.npy"), X_train)
    np.save(os.path.join(path, "Y_train.npy"), Y_train)
    np.save(os.path.join(path, "X_test.npy"), X_test)
    np.save(os.path.join(path, "Y_test.npy"), Y_test)

###################################

def loadDataset(path):
    """
    A function which gets numpy arrays of a dataset from binary files.
    """
    X_train = np.load(os.path.join(path, "X_train.npy"))
    Y_train = np.load(os.path.join(path, "Y_train.npy")).reshape(1, -1)
    X_test = np.load(os.path.join(path, "X_test.npy"))
    Y_test = np.load(os.path.join(path, "Y_test.npy")).reshape(1, -1)

    return X_train, Y_train, X_test, Y_test
