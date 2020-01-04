import os
import logging
import numpy as np
from keras.models import load_model
import datasets
from PIL import Image
from skimage import color
import random

def getSamples(root, pixel=32):
    X = []
    paths = []

    for file in os.listdir(root):
        imagePath = root + "/" + file

        try:
            image = Image.open(imagePath)
            image = np.asarray(image)
            image = np.array(Image.fromarray(image.astype('uint8')).resize((pixel, pixel)))

            if len(image.shape) == 2 or image.shape[2] == 1: # grayscale
                image = color.gray2rgb(image)
            elif image.shape[2] == 4:                        # alpha
                image = color.rgba2rgb(image)

            X.append(image)
            paths.append(imagePath)
        except:
            print(file, "could not be opened")

    return [np.asarray(X, dtype=np.float32)/255, paths]

###################################

def getImage(path, pixel=32):
    X = []

    try:
        image = Image.open(path)
        image = np.asarray(image)
        image = np.array(Image.fromarray(image.astype('uint8')).resize((pixel, pixel)))

        if len(image.shape) == 2 or image.shape[2] == 1: # grayscale
            image = color.gray2rgb(image)
        elif image.shape[2] == 4:                        # alpha
            image = color.rgba2rgb(image)

        X.append(image)
    except:
        print(path, "could not be opened")

    return np.asarray(X, dtype=np.float32)/255

###################################

def neuralNetwork(model, sample):
    logging.getLogger('tensorflow').disabled = True
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    np.set_printoptions(suppress=True) # ignoring scientific numeric notation
    #X_samples = getSamples("samples", pixel=32)
    #model = load_model("model.h5")
    #index = random.randint(0, X_samples.shape[0])
    #Y_pred = model.predict(X_samples[index:index+1])
    Y_pred = model.predict(sample)
    best_args = np.argsort(-Y_pred).T[:5]
    rates = -np.sort(-Y_pred).T[:5]

    return best_args, rates
