import math

import numpy as np


def b_v_sb(predictions):
    ret_val = []

    for prediction in predictions:
        idxs = np.argsort(prediction)
        ret_val.append(1 - prediction[idxs[0]] + prediction[idxs[1]])

    return np.argsort(ret_val)[::-1]


def ep_measure(predictions):
    ret_val = []

    for prediction in predictions:
        sum = 0
        for prob in prediction:
            if prob != 1 and prob != 0:
                sum += prob * math.log(prob)

        ret_val.append(-1 * sum)

    return np.argsort(ret_val)[::-1]


def bernoulli_max(predictions):
    ret_val = []

    for prediction in predictions:
        temp = []
        for prob in prediction:
            if prob == 1:
                temp.append(0)
            elif prob == 0:
                temp.append(0)
            else:
                temp.append(-1 * prob * math.log(prob) - (1 - prob) * math.log(1 - prob))
        ret_val.append(temp[np.argmax(temp)])

    return np.argsort(ret_val)[::-1]


def bernoulli_avg(predictions):
    ret_val = []

    for prediction in predictions:
        temp = []
        for prob in prediction:
            if prob == 1:
                temp.append(0)
            elif prob == 0:
                temp.append(0)
            else:
                temp.append(-1 * prob * math.log(prob) - (1 - prob) * math.log(1 - prob))
        ret_val.append(np.average(temp))

    return np.argsort(ret_val)[::-1]


def append_set(to_append, append_with, count):
    for i in range(0, count):
        to_append.append(append_with[i])
