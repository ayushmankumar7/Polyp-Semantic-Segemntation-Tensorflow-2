import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2 
from glob import glob
import tensorflow as tf 


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard

from data_prep import load_data, tf_dataset
from model import build_model


if __name__ == "__main__":

    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data("dataset/")

    print(len(train_x), len(test_x))

    