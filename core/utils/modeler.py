import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from core.tasks.classifier import Classifier  

class Modeler:
    def __init__(self, configs):
        self.model_name = configs.model_name
        self.task_type = configs.task_type
        self.num_class = configs.num_class
        self.input_shape = configs.input_shape
        self.input_h = configs.input_shape[0]
        self.input_w = configs.input_shape[1]
        self.input_c = configs.input_shape[2]
        self.optimizer = configs.optimizer
        self.loss = configs.loss
        self.metrics = configs.metrics
    
    def make_model(self,):
        inputs = Input(shape=(self.input_h, self.input_w, self.input_c))
    
        if self.task_type == "classification":
            assert self.num_class is not None, 'Classification task needs number of classes, check your config file'
            model = Classifier(self.model_name, self.input_shape, self.num_class)
        else:
            pass    
        self.modeler_model = model
        print(self.modeler_model.summary())
        
    def compile_model(self,):
        if self.task_type == "classification":
            self.modeler_model.compile(optimizer=self.optimizer,
                                       loss=self.loss,
                                       metrics=[self.metrics])
        else:
            pass
                                   
    def train_model(self, train_ds, batch_size, epochs):
        if self.task_type == "classification":
            self.modeler_model.fit(train_ds, batch_size=batch_size, epochs=epochs)
        else:
            pass
     
if __name__ == "__main__":
    pass