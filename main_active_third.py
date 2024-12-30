import os
import copy

import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from tensorflow.python.keras.callbacks import ModelCheckpoint

import utils.active_utils as au
import utils.cnn_utils as cu

labele = ["samoyed", "leonberg", "basset", "persian", "mau", "siamese"]


def train_cnn(j=0):
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
    x_remaining = x_train.astype('float32')
    x_remaining /= 255

    y_remaining = to_categorical(np.array(labels))

    x_remaining, x_total, y_remaining, y_total = train_test_split(x_remaining, y_remaining, test_size=1 / 5,
                                                                  random_state=42)
    sizes = [175, 175, 150, 150, 100]

    x_remaining_total = copy.deepcopy(x_remaining)
    y_remaining_total = copy.deepcopy(y_remaining)
    x_total_old = copy.deepcopy(x_total)
    y_total_old = copy.deepcopy(y_total)

    for metric in ["bvsb"]:  #"bvsb", "ep", "bmax", "bavg"
        cnn = cu.create_cnn(3)

        x_remaining = copy.deepcopy(x_remaining_total)
        y_remaining = copy.deepcopy(y_remaining_total)
        x_total = copy.deepcopy(x_total_old)
        y_total = copy.deepcopy(y_total_old)

        for i in range(0, 3):
            x_train, x_valid, y_train, y_valid = train_test_split(x_total, y_total, test_size=0.1, random_state=42)
            print('Train shape: ', x_train.shape, y_train.shape)
            print('Test shape: ', x_valid.shape, y_valid.shape)

            checkpointer = ModelCheckpoint(
                filepath="models_active/third/" + metric + "/cnn_active_" + str(i + 1) + "_third_" + str(j) + "_best.hdf5",
                verbose=1, save_best_only=True)

            cnn.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=30 - i * 5, batch_size=32,
                    callbacks=[checkpointer],
                    verbose=1, shuffle=True)

            # ucitamo i evaluiramo najbolji model zbog dobijanja statistike
            cnn = load_model("models_active/third/" + metric + "/cnn_active_" + str(i + 1) + "_third_" + str(j) + "_best.hdf5")
            loss, acc = cnn.evaluate(x_valid, y_valid, verbose=1)

            with open('models_active/third/active_' + metric + '_results_' + str(j) + '_best.txt', 'a') as f:
                f.write('iter: ' + str(i + 1) + '\n')
                f.write('val_loss: ' + str(loss) + '\n')
                f.write('val_accuracy: ' + str(acc) + '\n')
                f.write('---------------------------\n')

            del x_train
            del y_train
            del x_valid
            del y_valid

            # izracunamo metriku
            if metric == "bvsb":
                idxs = au.b_v_sb(cnn.predict(x_remaining))
            elif metric == "ep":
                idxs = au.ep_measure(cnn.predict(x_remaining))
            elif metric == "bmax":
                idxs = au.bernoulli_max(cnn.predict(x_remaining))
            else:
                idxs = au.bernoulli_avg(cnn.predict(x_remaining))

            # sortiramo po metrici
            x_remaining = x_remaining[idxs]
            y_remaining = y_remaining[idxs]

            # dodamo najboljih n u trening skup
            x_total = np.concatenate([x_total, x_remaining[0:(sizes[i] * 3)]])
            y_total = np.concatenate([y_total, y_remaining[0:(sizes[i] * 3)]])

            # izbacimo najboljih n iz ostatka
            x_remaining = x_remaining[(sizes[i] * 3):]
            y_remaining = y_remaining[(sizes[i] * 3):]


if __name__ == '__main__':
    for n in [0, 1, 2]:
        train_cnn(n)
    exit(0)
