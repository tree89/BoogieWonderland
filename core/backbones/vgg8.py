import os
os.environ['TF2_BEHAVIOR']='1'
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation, Input, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.models import Model


def vgg_block(x, filters, layers, name, weight_decay):
    for i in range(layers):
        x = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=regularizers.l2(weight_decay), name=f'{name}_conv_{i}')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    return x


def Vgg8(x, weight_decay=1e-4, is_classifier=False):
    

    x = vgg_block(x, 16, 2, 'block_1', weight_decay)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = vgg_block(x, 32, 2, 'block_2', weight_decay)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = vgg_block(x, 64, 2, 'block_3', weight_decay)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    if is_classifier:
        x = Dropout(0.5)(x)
        x = Flatten()(x)
        x = Dense(512, kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(weight_decay), name='dense_1')(x)

    return x


if __name__ == "__main__":
    inputs = Input((224, 224, 3))
    vgg8 = Model(inputs, Vgg8(inputs, 10, True))
    print(vgg8.summary())
