import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from core.backbones.Alexnet import Alexnet

def load_trainer(model_name, task_type, input_shape=[224, 224, 3]):
    inputs = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    model_dict = {"Alexnet" : Alexnet(inputs, True)}
    model = model_dict[model_name]
    Base_model = Model(inputs, model)
    
    return Base_model
"""
class trainer(self):
    def __init__(self, things):
        pass
"""
     
if __name__ == "__main__":
    model = load_trainer("Alexnet", "classification", [224,224,3])
    print(model.summary())