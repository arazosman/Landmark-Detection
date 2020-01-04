# pylint: disable = line-too-long, too-many-lines, too-many-arguments, wrong-import-order, invalid-name, missing-docstring, redefined-outer-name

import os
import logging
import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, Flatten, Dense, BatchNormalization, Activation, Dropout
from keras.models import Model, Sequential
# from sklearn.model_selection import train_test_split
from keras.regularizers import l2
from keras.utils import plot_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import datasets

def getModel(input_shape, C):
    X_input = Input(input_shape)

    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1))(X_input)
    X = BatchNormalization(axis=3)(X)
    X = Activation("relu")(X)

    X = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation("relu")(X)
    X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(X)

    X = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation("relu")(X)

    X = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation("relu")(X)
    X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(X)

    X = Flatten()(X)
    X = Dense(512, activation="relu")(X)
    X = Dropout(0.5)(X)
    X = Dense(C, activation="softmax")(X)

    model = Model(inputs=X_input, outputs=X)

    return model

###################################

logging.getLogger('tensorflow').disabled = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

###################################

classes = datasets.getClasses("sub-datasets/bridges")
C = len(classes)
print(C)

X_train, Y_train, X_test, Y_test = datasets.getDataset("sub-datasets/bridges", classes, pixel=32, rate=.85)

X_train = X_train / 255 # normalizing
Y_train = np.eye(C)[Y_train.reshape(-1)] # one-to-hot matrix
X_test = X_test / 255 # normalizing
Y_test = np.eye(C)[Y_test.reshape(-1)] # # one-to-hot matrix

print("Dataset dimensions:", X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

model = getModel(X_train.shape[1:], C)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(x=X_train, y=Y_train, validation_split=.15, epochs=20, batch_size=32)

preds = model.evaluate(x=X_test, y=Y_test)

print()
print("Loss: " + str(preds[0]))
print("Test Accuracy: " + str(preds[1]))

model.summary()
plot_model(model, to_file='model.png')

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('history.png')

print(classes)

y_pred = model.predict(X_test)
matrix = confusion_matrix(Y_test.argmax(axis=1), y_pred.argmax(axis=1))
np.savetxt("confusion_matrix.txt", matrix)

