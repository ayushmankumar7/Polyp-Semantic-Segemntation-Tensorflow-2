import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2 
from glob import glob
import tensorflow as tf 

from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard

from data_prep import load_data, tf_dataset
from model import build_model


def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x

    return tf.numpy_function(f, [y_true, y_pred], tf.float32)


if __name__ == "__main__":

    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data("dataset/")

    print(len(train_x), len(test_x))

    
    batch = 8
    lr = 1e-4
    epochs = 20

    train_dataset = tf_dataset(train_x, train_y, batch= batch)
    train_dataset = tf_dataset(valid_x, valid_y, batch= batch)  

    model = build_model()

    optimizer = tf.keras.optimizer.Adam(lr)
    mertics = ['acc', Recall(), Precision(), iou]

    model.compile(loss = "binary_crossentropy", optimizer = optimizer, metrics = metrics)

    callbacks = [
        ModelCheckpoint("files/model.h5"),
        ReduceLROnPlateau(monitor ="val_loss", factot =0.1, patience = 3),
        CSVLogger("files/data.csv").
        TensorBoard(),
        EarlyStopping(monitor = "val_loss", patience = 10, restore_best_weights = False)
        
    ]
