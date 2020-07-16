import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2 
from glob import glob
import tensorflow as tf 


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)



from tqdm import tqdm
from tensorflow.keras.utils import CustomObjectScope
from data_prep import load_data, tf_dataset
from train import iou



def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (256, 256))
    x = x/255.0
    return x

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (256, 256))
    x = x/255.0
    x = np.expand_dims(x, axis=-1)
    return x

def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask,mask,mask]
    mask = np.transpose(mask, (1,2,0))
    return mask

if __name__ == "__main__":
    
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data("dataset/")
    print(len(train_x), len(test_x))
    batch = 8
    
    test_dataset = tf_dataset(test_x, test_y, batch = batch)
    test_steps = len(test_x)//batch

    if len(test_x) % batch != 0:
        test_steps +=1

    with CustomObjectScope({'iou':iou}):
        model = tf.keras.models.load_model("files/model.h5")

    model.evaluate(test_dataset, steps = test_steps)

    for i , (x,y) in tqdm(enumerate(zip(text_x, test_y)), total = len(text_x)):
        x = read_img(x)
        y = read_mask(y)
        y_pred = model.predict(np.expand_dims(x, axis = 1))
        y_pred = y_pred[0] > 0.5 
        h,w,_ = x.shape

        white_line = np.ones((h, 5, 3)) * 255.0
        all_images =[
            x = 255.0,
            white_line,
            mask_parse(y).
            white_line,
            mask_parse(y_pred) * 255.0
        ]

        image = np.concatenate(all_images, axis = 1)
        cv2.imwrite(f"results/{i}.png", image)



