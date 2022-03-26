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
model_name = "gssp3dNet"


def main():
    model_path = "/mnt/hdd/liucheng/gcp3d_ablation/models/gssp3dNet_0.8682_0.0286/{}_fold_{}.h5".format(model_name, 1)
    with CustomObjectScope({'f1_score': f1_score, 'AUC': AUC,
                        'recall': recall, 'precision': precision}):
        model = load_model(model_path)
        model.summary()    
        


if __name__ == '__main__':
    main()
