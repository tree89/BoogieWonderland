import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from core.tasks.classifier import Classifier  

class Modeler:
    def __init__(self, model_name, task_type, num_class=None, input_shape=[224, 224, 3]):
        self.model_name = model_name
        self.task_type = task_type
        self.num_class = num_class
        self.input_shape = input_shape  
    
    def make_model(self,):
        inputs = Input(shape=(self.input_shape[0], self.input_shape[1], self.input_shape[2]))
    
        if self.task_type == "classification":
            assert self.num_class is not None, 'Classification task needs number of classes, check your config file'
            model = Classifier(self.model_name, self.input_shape, self.num_class)
        else:
            pass    
        self.modeler_model = model
        
        return self.modeler_model
     
if __name__ == "__main__":
    modeler = Modeler("vgg8", "classification", 10, [32,32,3])
    model = modeler.make_model()
    print(model.summary())