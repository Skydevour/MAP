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
from metrics import f1_score, AUC, recall, precision
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
from keras.utils import CustomObjectScope
import cv2
from gss_gcp3d_resnet import Resnet3DBuilder
from keras.utils import multi_gpu_model
import nibabel as nib
from tqdm import tqdm
import pickle


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Global Constants
nb_class = 2

# Xception, InceptionResNetV2, InceptionV3
IM_HEIGHT = 224
IM_WIDTH = 224
model_name = "gcp3d"

dataPath = "./data"

batchsize = 1
EPOCH = 50
init_lr = 1e-3


def load_data():
    datas = []
    labels = []
    for class_dir in os.listdir(dataPath):
        classPath = os.path.join(dataPath, class_dir)
        if "MCI" in classPath:
            continue
        for idx, idm in tqdm(enumerate(os.listdir(classPath))):
            lbl = 1 if "AD" in classPath else 0
            labels.append(lbl)
            file = os.path.join(classPath, idm)
            data = np.array(nib.load(file).get_data())
            data = data / data.max()
            data = cv2.resize(data, (IM_HEIGHT, IM_WIDTH))
            datas.append(data)
    labels = np.asarray(labels)
    return datas, labels


def plot_roc(fpr, tpr, value, fold):
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
    plt.savefig("results/roc_auc/{}_{}_roc_curve.png".format(model_name, fold))



def lr_scheduler(epoch):
    global init_lr
    lr = init_lr
    if 5 <= epoch < 10:
        lr = init_lr * 0.5
    elif 10 <= epoch < 20:
        lr = init_lr * 0.1
    elif 20 <= epoch < EPOCH:
        lr = init_lr * 0.05
    #elif 60 <= epoch:
    #    lr = init_lr * 0.01
    print("Learning rate: %.6f" % lr)
    return lr



def eval(x_val, y_val, model_path, fold):
    with CustomObjectScope({'f1_score': f1_score, 'AUC': AUC,
                            'recall': recall, 'precision': precision}):
        model = load_model(model_path)

    print("-------- Evaluated Result --------")

    y_pred = model.predict(x_val, batch_size=batchsize)
    y_pred_bool = np.argmax(y_pred, axis=1)
    y_val_hot = np_utils.to_categorical(y_val, nb_class)
    fpr, tpr, thresholds = roc_curve(y_val_hot.ravel(), y_pred.ravel())
    value = auc(fpr, tpr)
    plot_roc(fpr, tpr, value, fold)
    print("Accuracy: %.4f" % accuracy_score(y_val, y_pred_bool))
    print("Micro Auc: %.4f" % value)
    print(classification_report(y_val, y_pred_bool, digits=4))



def build_model(img_size):
    model = Resnet3DBuilder.build_resnet_50(img_size, nb_class)
    opt = Adam(init_lr)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc', f1_score, AUC, recall, precision])
    print ('[INFO] Model Compiled')
    return model



def save_metric(hist, savePath):
    with open(savePath, 'wb') as file:
        pickle.dump(hist.history, file)



def main():
    x, y = load_data()
    x = np.expand_dims(x, axis=-1)
    print("[INFO] Data shape: {}".format(x.shape))  # (300, 79, 95, 79)
    print("[INFO] Label shape: {}".format(y.shape))  # (300,)

    skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    for fold, (train_index, val_index) in enumerate(skf.split(x, y)):
        #if fold != 4:
            #continue
        print("[INFO] Training on fold {}".format(fold+1))
        x_train, y_train = x[train_index], y[train_index]
        x_val, y_val = x[val_index], y[val_index]

        y_train_hot = np_utils.to_categorical(y_train, nb_class)
        y_val_hot = np_utils.to_categorical(y_val, nb_class)

        model_path = "models/tmp/{}_fold_{}.h5".format(model_name, fold + 1)

        savePath = "results/history/tmp/{}_fold_{}_history.pckl".format(model_name, fold + 1)
        checkpoint = ModelCheckpoint(model_path, monitor='val_acc', verbose=1,
                                     save_best_only=True, mode='max')
        reduce_lr = LearningRateScheduler(lr_scheduler)

        model = build_model((x.shape[1], x.shape[2], x.shape[3], 1))

        hist = model.fit(x_train, y_train_hot, batch_size=batchsize, epochs=EPOCH,
                         validation_data=(x_val, y_val_hot), shuffle=True,
                         callbacks=[reduce_lr, checkpoint])
        print("[INFO]Final LR: %s" % K.get_value(model.optimizer.lr))
        eval(x_val, y_val, model_path, fold + 1)
        save_metric(hist, savePath)

        del hist
        del model
        del x_train, y_train
        del x_val, y_val
        K.clear_session()


if __name__ == '__main__':
    main()
