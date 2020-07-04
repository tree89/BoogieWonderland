import os

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

from core.backbones.alexnet import Alexnet
from core.backbones.vgg8 import Vgg8



model_dict = {"Alexnet" : Alexnet, "vgg8":Vgg8}

def Classifier(model_name, input_shape, num_class):
    inputs = Input(input_shape)
    base_model = Model(inputs, model_dict[model_name](inputs, is_classifier=True))
    outputs = Dense(units=num_class, activation='softmax')(base_model.output)    
    classifier = Model(inputs, outputs)
    
    return classifier
    
if __name__ == "__main__":
    classifier = Classifier("vgg8", (32,32,3), 10)
    print(classifier.summary())