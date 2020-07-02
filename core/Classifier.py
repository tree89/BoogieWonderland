import os

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation, Input, MaxPooling2D, Dropout, Flatten, Input
from tensorflow.keras.models import Model

from core.backbones.Alexnet import *

#def Classifier(model_name, input_shaep, num_class):
    
inputs = Input(shape=(224, 224, 3))
Base_model = Model(inputs, Alexnet(inputs, True))
print(Base_model.summary())

#if __name__ == "__main__":
#    inputs = Input(shape=(224, 224, 3))
#    Base_model = Model(inputs, Alex(inputs, True))
#    print(Base_model.summary())
