import os

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation, Input, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.models import Model

# build model layers
def Conv(x, filters, kernel_size, strides, padding='same', activation='relu'):
    x = Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size), strides=strides, padding=padding,
               kernel_initializer=tf.random_normal_initializer(0., 0.02))(x)
    x = Activation(activation)(x)
    return x


def Alexnet(x, is_classifier=False):
    
    x = Conv(x, 96, 11, strides=4, padding='valid')
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(x)

    x = Conv(x, 256, 5, strides=1, padding='same')
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(x)

    x = Conv(x, 384, 3, strides=1, padding='same')
    x = Conv(x, 384, 3, strides=1, padding='same')
    x = Conv(x, 256, 3, strides=1, padding='same')
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(x)

    if is_classifier:

        x = Flatten()(x)
        x = Dense(4096, activation='relu', use_bias=True)(x)
        x = Dropout(0.5)(x)
        x = Dense(4096, activation='relu', use_bias=True)(x)
        x = Dropout(0.5)(x)

    return x


if __name__ == "__main__":
    inputs = Input(shape=(224, 224, 3))
    Base_model = Model(inputs, Alexnet(inputs, True))
    print(Base_model.summary())
