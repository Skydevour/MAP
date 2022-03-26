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
from metrics import f1_score, AUC
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
from keras.utils import CustomObjectScope
import cv2
from gss_dense_net import gssdNet40
from keras.utils import multi_gpu_model
import nibabel as nib
from tqdm import tqdm
import pickle



os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)


#f = h5py.File(r'./data/train/train_pre_data.h5', 'r')
#y = pd.read_csv(r'./data/train/train_pre_label.csv')

# Global Constants
nb_class = 2

# Xception, InceptionResNetV2, InceptionV3
IM_HEIGHT = 224
IM_WIDTH = 224
model_name = "gssdNet"

'''
# VGG16, DenseNet121
IM_HEIGHT = 224
IM_WIDTH = 224
'''

dataPath = "data"

batchsize = 16
EPOCH = 60
init_lr = 1e-3


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
            datas.append(data)
    #datas = itensity_normalize_one_volume(np.asarray(datas))
    labels = np.asarray(labels)
    return datas, labels


def resize_data(data, label):
    ret = []
    lbl = []
    for idx in range(0, len(data)):   # (num, height, width, depth)
        subject = data[idx,:,:,:]
        subject = cv2.resize(subject, (IM_HEIGHT, IM_WIDTH), cv2.INTER_CUBIC)
        ret.append(subject)
        lbl.append(label[idx])
    ret = np.asarray(ret)
    lbl = np.asarray(lbl)
    return ret, lbl


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
    plt.savefig("{}_{}_roc_curve.png".format(model_name, fold))



def lr_scheduler(epoch):
    ratio = float(epoch) / EPOCH
    pr = 0.9
    lr = init_lr * np.power((1 - ratio), pr)
    print("Learning rate: %.6f" % lr)
    return lr



def eval(x_val, y_val, model_path, fold):
    with CustomObjectScope({'f1_score': f1_score, 'AUC': AUC}):
        model = load_model(model_path)

    #y_val_hot = np_utils.to_categorical(y_val, nb_class)
    #score = model.evaluate(x_val, y_val_hot, batch_size=batchsize) # [loss, acc, f1]
    print("-------- Evaluated Result --------")
    #print("%s: %.2f%%\n%s:%.2f%%" %
    #      (model.metrics_names[1], score[1] * 100, model.metrics_names[2], score[2] * 100))

    y_pred = model.predict(x_val, batch_size=batchsize)
    y_pred_bool = np.argmax(y_pred, axis=1)
    fpr, tpr, thresholds = roc_curve(y_val, y_pred[:, 1], pos_label=1, drop_intermediate=False)
    value = auc(fpr, tpr)
    plot_roc(fpr, tpr, value, fold)
    print("Accuracy: %.2f" % accuracy_score(y_val, y_pred_bool))
    print("Auc: %.2f" % value)
    print(classification_report(y_val, y_pred_bool))



def build_model(img_size):
    '''
    conv_base = InceptionV3(include_top=None, weights=None, input_shape=img_size)
    inpt = conv_base.input
    x = conv_base.output
    x = AveragePooling2D(K.int_shape(x)[1:-1])(x)
    x = Flatten()(x)
    pred = Dense(nb_class, activation='softmax')(x)
    model = Model(inputs=inpt, outputs=pred)
    '''
    model = gssdNet40(img_size, nb_class)
    #para_model = multi_gpu_model(model, gpus=2)
    #opt = Nadam(init_lr, beta_1=0.9, beta_2=0.999)
    opt = Adam(init_lr)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc', f1_score, AUC])
    print ('[INFO] Model Compiled')
    return model


def save_metric(hist, savePath):
    with open(savePath, 'wb') as file:
        pickle.dump(hist.history, file)


def getClassw(y):
    num_per_class = np.bincount(y)
    min_class = float(num_per_class[np.argmin(num_per_class)])
    class_w = {0: min_class/num_per_class[0], 1:min_class/num_per_class[1]}
    return class_w



def main():
    x, y = load_data()
    #x, y = resize_data(x, y)
    x = np.expand_dims(x, axis=-1)
    print("[INFO] Data shape: {}".format(x.shape))  # (300, 79, 95, 79)
    print("[INFO] Label shape: {}".format(y.shape))  # (300,)

    #x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=0)


    skf = StratifiedKFold(n_splits=10, random_state=2020, shuffle=False)
    for fold, (train_index, val_index) in enumerate(skf.split(x, y)):
        x_train, y_train = x[train_index], y[train_index]
        x_val, y_val = x[val_index], y[val_index]

        y_train_hot = np_utils.to_categorical(y_train, nb_class)
        y_val_hot = np_utils.to_categorical(y_val, nb_class)

        class_w = getClassw(y)
        print("Class Weight: %s" % class_w)
        model_path = "models/ad_vs_cn/fold_{}_{}.h5".format(fold + 1, model_name)
        savePath = "results/{}_fold_{}_history.pckl".format(model_name, fold + 1)
        #checkpoint = ModelCheckpoint(model_path, monitor='val_acc', verbose=1,
        #                             save_best_only=True, mode='max')
        reduce_lr = LearningRateScheduler(lr_scheduler)
        #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
        #                              patience=5, min_delta=0.001)

        model = build_model((x.shape[1], x.shape[2], x.shape[3], 1))
        model.summary()
        hist = model.fit(x_train, y_train_hot, batch_size=batchsize, epochs=EPOCH,
                         validation_data=(x_val, y_val_hot), shuffle=True,
                         callbacks=[reduce_lr], class_weight=class_w)
        print("[INFO]Final LR: %s" % K.get_value(model.optimizer.lr))
        model.save(model_path)
        eval(x_val, y_val, model_path, fold + 1)
        save_metric(hist, savePath)
        break


if __name__ == '__main__':
    main()
