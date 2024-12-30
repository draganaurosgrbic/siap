from tensorflow import keras

import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

best_size = (224, 224)


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)


def load_image_color(path):
    return cv2.imread(path)


def load_image_color_rgb(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def display_image(image):
    plt.imshow(image)


# transformisemo u oblik pogodan za scikit-learn
def reshape_data(input_data):
    nsamples, nx, ny = input_data.shape
    return input_data.reshape((nsamples, nx * ny))


def create_cnn(class_num=3):
    cnn3 = Sequential()

    cnn3.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)))
    cnn3.add(MaxPooling2D((2, 2)))
    cnn3.add(Dropout(0.25))

    cnn3.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    cnn3.add(MaxPooling2D(pool_size=(2, 2)))
    cnn3.add(Dropout(0.25))

    cnn3.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    cnn3.add(MaxPooling2D(pool_size=(2, 2)))
    cnn3.add(Dropout(0.25))

    cnn3.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    cnn3.add(MaxPooling2D(pool_size=(2, 2)))
    cnn3.add(Dropout(0.25))

    cnn3.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    cnn3.add(MaxPooling2D(pool_size=(2, 2)))
    cnn3.add(Dropout(0.25))

    cnn3.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    cnn3.add(MaxPooling2D(pool_size=(2, 2)))
    cnn3.add(Dropout(0.25))

    cnn3.add(Flatten())

    cnn3.add(Dense(256, activation='relu'))
    cnn3.add(Dropout(0.5))

    cnn3.add(Dense(class_num, activation='softmax'))

    cnn3.compile(loss=keras.losses.categorical_crossentropy,
                 optimizer=keras.optimizers.Adam(),
                 metrics=['accuracy'])
    return cnn3


def classify_rect(image, clf_cnn):
    image = np.array([image])
    image = image.reshape(image.shape[0], 224, 224, 3)
    image = image.astype('float32')
    image /= 255

    prediction = clf_cnn.predict(image)
    return np.max(prediction), np.argmax(prediction, axis=1)[0]


def process_whole_image(image, clf_cnn):
    image = cv2.resize(image, best_size, interpolation=cv2.INTER_CUBIC)
    prediction = classify_rect(image, clf_cnn)
    return prediction[0], prediction[1]