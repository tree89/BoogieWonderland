# PYTHONPATH=. python main.py --config=configs/classification_test.json

import os
import json
import argparse
import tensorflow as tf

from core.backbones import *
from core.utils.data_loader import load_data
from core.utils.modeler import Modeler  

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True, help="Path of configs file")

args = vars(ap.parse_args())

config_path = args["config"]

f = open(config_path, 'r')
configs = json.load(f)

model_name = configs["model_name"]
task_type = configs["task"]
num_class = configs["num_class"]
dataset = configs["dataset"]
batch_size = configs["batch_size"]
epoch = configs["epochs"]
input_shape = configs["input_shape"]
h = input_shape[0]
w = input_shape[1]
c = input_shape[2]
save_path = configs["save_path"]
load_model = configs["load_model"]

if not(os.path.isdir(save_path)):
    os.makedirs(os.path.join(save_path))

model_dict = {"Alexnet", "vgg8"}
task_name = {"classification", "detection"}
dataset_name = {"cifar10"}


def main():
    assert model_name in model_dict, f'This repository don\'t support {model_name}' 
    assert task_type in task_name, f'{task_type} is wrong task type'
    assert dataset in dataset_name, f'This repository don\'t support {dataset}'
    assert batch_size > 0, 'Batch size must be lager than 1'
    assert epoch > 0, 'Epoch must be lager than 1'
    
    train_ds, val_ds = load_data(dataset, batch_size)
    load_model = False
    if load_model:
        model = tf.keras.models.load_model(load_model)
        print(model.summary())
    else:
        modeler = Modeler(model_name, task_type, num_class, [h, w, c])
        model = modeler.make_model()  
        print(model.summary())
        
    if task_type == "classification":
        model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
        model.fit(train_ds, batch_size=batch_size, epochs = epoch)

if __name__ == "__main__":
    main()