import os
import time

import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from tensorflow.python.keras.callbacks import ModelCheckpoint
import utils.cnn_utils as cu


labele = ["persian", "mau", "siamese"]
img_w = 0
img_h = 0

train_dir_neg = 'dogs/train/neg/'
test_dir = 'dogs/test/'

pos_imgs = []
neg_imgs = []
test_imgs = []


def train_cnn():
    imgs = []
    labels = []

    temp = []
    temp_labels = []
    for img_name in os.listdir("train/siamese/"):
        img_path = os.path.join("train/siamese/", img_name)
        img = cu.load_image_color(img_path)
        img = cv2.resize(img, cu.best_size, interpolation=cv2.INTER_CUBIC)
        temp.append(img)
        temp_labels.append(2)
    imgs.extend(temp)
    labels.extend(temp_labels)

    temp = []
    temp_labels = []
    for img_name in os.listdir("train/mau/"):
        img_path = os.path.join("train/mau/", img_name)
        img = cu.load_image_color(img_path)
        img = cv2.resize(img, cu.best_size, interpolation=cv2.INTER_CUBIC)
        temp.append(img)
        temp_labels.append(1)
    imgs.extend(temp)
    labels.extend(temp_labels)

    temp = []
    temp_labels = []
    for img_name in os.listdir("train/persian/"):
        img_path = os.path.join("train/persian/", img_name)
        img = cu.load_image_color(img_path)
        img = cv2.resize(img, cu.best_size, interpolation=cv2.INTER_CUBIC)
        imgs.append(img)
        labels.append(0)
    imgs.extend(temp)
    labels.extend(temp_labels)

    imgs = np.array(imgs)
    labels = np.array(labels)
    print(imgs.shape)

    x_train = imgs.reshape(imgs.shape[0], 224, 224, 3)
    x_train = x_train.astype('float32')
    x_train /= 255

    y_train = to_categorical(np.array(labels))
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
    print('Train shape: ', x_train.shape, y_train.shape)
    print('Test shape: ', x_valid.shape, y_valid.shape)
    cnn = cu.create_cnn(3)

    checkpointer = ModelCheckpoint(filepath="cnn_best_third.hdf5", verbose=1, save_best_only=True)

    cnn.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=30, batch_size=32, callbacks=[checkpointer],
            verbose=1, shuffle=True)
    cnn.save("cnn_third.hdf5")


if __name__ == '__main__':
    # train_cnn()
    # exit(0)
    clf_cnn = load_model("models/cnn_third_03178_9936.hdf5")

    start = time.time()
    for img_name in os.listdir("test2"):
        img_path = os.path.join("test2/", img_name)
        itest = cu.load_image_color(img_path)
        (img_h, img_w) = itest.shape[:2]

        scores, classes = cu.process_whole_image(itest, clf_cnn)
        cv2.putText(itest, labele[classes] + "{:6.3f}".format(scores),
                    (0 + 5, 0 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (36, 255, 12), 2)
        cv2.rectangle(itest, (0, 0), (img_w, img_h), (0, 255, 0), 2)

        cv2.imwrite('output/cnn/combo/whole/third/' + img_name.split('.')[0] + '.png', itest)

    end = time.time()
    total = end - start
    print(total)
    print(total / 420)
