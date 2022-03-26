#! /usr/bin python
# coding:utf-8

import matplotlib
matplotlib.use("Agg") 
from keras.models import Model, load_model, Sequential
from keras.layers import Conv2D, Activation, Dense
from keras.layers import Flatten, Input, AveragePooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.optimizers import Adam, Nadam
from keras.utils import np_utils
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
import keras.backend as K
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import h5py
import pandas as pd
from metrics import f1_score
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
from keras.utils import CustomObjectScope
import cv2
from resnet3d import Resnet3DBuilder
from keras.utils import multi_gpu_model
import pickle



os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)


#f = h5py.File(r'./data/train/train_pre_data.h5', 'r')
#y = pd.read_csv(r'./data/train/train_pre_label.csv')

# Global Constants
nb_class = 3
channels = 3

# Xception, InceptionResNetV2, InceptionV3
IM_HEIGHT = 224
IM_WIDTH = 224

'''
# VGG16, DenseNet121
IM_HEIGHT = 224
IM_WIDTH = 224
'''

dataPath = "./data/train/train_pre_data.h5"
labelPath = "./data/train/train_pre_label.csv"
model_name = "3D_resnet18"
model_path = "models/{}.h5".format(model_name)

batchsize = 8
EPOCH = 150
init_lr = 1e-2


def rot_data(data, label):
    ret = []
    lbl = []
    for idx in range(0, len(data)):   # (num, height, width, depth)
        subject = data[idx,:,:,:]
        #if idx in train_rot_idx:  # need to rotate
           #subject = np.transpose(subject, (2,1,0))
        #    continue
        subject = cv2.resize(subject, (IM_HEIGHT, IM_WIDTH), cv2.INTER_CUBIC)
        ret.append(subject)
        lbl.append(label[idx])
    ret = np.asarray(ret)
    lbl = np.asarray(lbl)
    return ret, lbl


def itensity_normalize_one_volume(volume):
    """
    normalize the itensity of an nd volume based on the mean and std of nonzeor region
    inputs:
        volume: the input nd volume
    outputs:
        out: the normalized nd volume
    """
    volume = volume.astype(np.float32)
    pixels = volume[volume > 0]
    mean = pixels.mean()
    std = pixels.std()
    out = (volume - mean) / std
    out_random = np.random.normal(0, 1, size=volume.shape)
    out[volume == 0] = out_random[volume == 0]
    return out


def load_data():
    # load data
    f = h5py.File(dataPath, 'r')
    x = np.array(f['data'])
    x = x[:, 0, :, :, :]

    # load label
    y = pd.read_csv(labelPath)
    y = np.array(y['label'])

    x = itensity_normalize_one_volume(x)  # normalization
    x, y = rot_data(x, y)      # rotate partial data
    #x = x / x.max()

    return x, y


def lr_scheduler(epoch):
    lr = init_lr
    if epoch >= 20:
        lr = 1e-3
    print("Learning rate: %s" % lr)
    return lr


def plot_roc(fpr, tpr, value):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % value)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig("results/roc_auc/{}_roc_curve.png".format(model_name))


def eval(x_val, y_val):
    with CustomObjectScope({'f1_score': f1_score}):
        model = load_model(model_path)

    #y_val_hot = np_utils.to_categorical(y_val, nb_class)
    #score = model.evaluate(x_val, y_val_hot, batch_size=batchsize) # [loss, acc, f1]
    print("-------- Evaluated Result --------")
    #print("%s: %.2f%%\n%s:%.2f%%" %
    #      (model.metrics_names[1], score[1] * 100, model.metrics_names[2], score[2] * 100))

    y_pred = model.predict(x_val, batch_size=batchsize)
    y_pred_bool = np.argmax(y_pred, axis=1)
    y_val_hot = np_utils.to_categorical(y_val, nb_class)
    fpr, tpr, thresholds = roc_curve(y_val_hot.ravel(), y_pred.ravel())
    value = auc(fpr, tpr)
    plot_roc(fpr, tpr, value)
    print("Accuracy: %.4f" % accuracy_score(y_val, y_pred_bool))
    print("Micro Auc: %.4f" % value)
    print(classification_report(y_val, y_pred_bool, digits=4))


def build_model(img_size):

    model = Resnet3DBuilder.build_resnet_18(img_size, nb_class)
    model = multi_gpu_model(model, gpus=2)
    #opt = Nadam(init_lr, beta_1=0.9, beta_2=0.999)
    opt = Adam(init_lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc', f1_score])
    print ('[INFO] Model Compiled')
    return model


def getClassw(y):
    num_per_class = np.bincount(y)
    min_class = float(num_per_class[np.argmin(num_per_class)])
    class_w = {0: min_class/num_per_class[0], 1:min_class/num_per_class[1],
               2: min_class/num_per_class[2]}
    return class_w


def pix2vox(x):
    data = np.zeros((x.shape[0], x.shape[1], x.shape[2], x.shape[3], 1), dtype='float32')
    for idx in range(x.shape[0]):
        data[idx, :, :, :, 0] = x[idx]
    return data


def save_metric(hist, savePath):
    with open(savePath, 'wb') as file:
        pickle.dump(hist.history, file)


def main():
    x, y = load_data()
    x = pix2vox(x)
    print("[INFO] Data shape: {}".format(x.shape))  # (300, 79, 95, 79)
    print("[INFO] Label shape: {}".format(y.shape))  # (300,)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=0)

    class_w = getClassw(y)

    # class_w = {0: 1., 1: 0.4503, 2:0.8395}

    # (300, 79, 95, 79) > (300*79, 79, 95, 3) | (300,) > (300*79,)
    # x_train, y_train = select_all_slices(x_train, y_train)
    # x_val, y_val = select_all_slices(x_val, y_val)

    y_train_hot = np_utils.to_categorical(y_train, nb_class)  # (300,) > (300, 3)
    y_val_hot = np_utils.to_categorical(y_val, nb_class)

    checkpoint = ModelCheckpoint(model_path, monitor='val_f1_score', verbose=1,
                                 save_best_only=True, mode='max')
    #reduce_lr = LearningRateScheduler(lr_scheduler)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=10, min_delta=0.001)

    model = build_model((x.shape[1], x.shape[2], x.shape[3], 1))
    savePath = "results/history/{}_history.pckl".format(model_name)
    #model.summary()
    hist = model.fit(x_train, y_train_hot, batch_size=batchsize, epochs=EPOCH,
              validation_data=(x_val, y_val_hot), shuffle=True,
              callbacks=[checkpoint, reduce_lr], class_weight=class_w)
    print("[INFO]Final LR: %s" % K.get_value(model.optimizer.lr))
    eval(x_val, y_val)
    save_metric(hist, savePath)



if __name__ == '__main__':
    main()
