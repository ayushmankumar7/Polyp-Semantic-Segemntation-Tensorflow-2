import tensorflow as tf 
from tensorflow.keras.layers import Conv2D, BatchNormalization,MaxPool2D, Activation, UpSampling2D, Input
from tensorflow.keras.models import Model


def conv_block(x, num_filters):
    x = Conv2D(num_filters, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(num_filters, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x



def build_model():
    size = 256
    num_filters = [16, 32, 48, 64]

    inputs = Input(shape = [size,size, 3])

    skip_x = []
    x = inputs

    #Encoder

    for f in num_filters:
        x = conv_block(x,f)
        skip_x.append(x)
        x = MaxPool2D((2,2))(x)
        
    #Bridge

    