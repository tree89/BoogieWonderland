import os
import json

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

model_dict = {"Alexnet", "vgg8"}
task_name = {"classification", "detection"}
dataset_name = {"cifar10"}


class Configurator:
    def __init__(self, config_path):
        self.config_path = config_path

    def load_configs(self, ):
        f = open(self.config_path, 'r')
        self.configs = json.load(f)
        self.model_name = self.configs["model_name"]
        self.task_type = self.configs["task"]
        self.num_class = self.configs["num_class"]
        self.dataset = self.configs["dataset"]
        self.batch_size = self.configs["batch_size"]
        self.epoch = self.configs["epochs"]
        self.input_shape = self.configs["input_shape"]
        self.input_h = self.input_shape[0]
        self.input_w = self.input_shape[1]
        self.input_c = self.input_shape[2]
        self.save_path = self.configs["save_path"]
        self.load_model = self.configs["load_model"]
        self.optimizer = self.configs["optimizer"]
        self.loss = self.configs["loss"]
        self.metrics = self.configs["metrics"]

        if not (os.path.isdir(self.save_path)):
            os.makedirs(os.path.join(self.save_path))

        return self.configs

    def check_configs(self, ):
        assert self.model_name in model_dict, f'This repository don\'t support {model_name}'
        assert self.task_type in task_name, f'{task_type} is wrong task type'
        assert self.dataset in dataset_name, f'This repository don\'t support {dataset}'
        assert self.batch_size > 0, 'Batch size must be lager than 1'
        assert self.epoch > 0, 'Epoch must be lager than 1'


if __name__ == "__main__":
    configurator = Configurator("/hdd2/home/cbpark/BoogieWonderland/configs/classification_test.json")
    configs = configurator.load_configs()
    print(configs)
