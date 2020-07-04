# PYTHONPATH=. python main.py --backbone Alexnet --task classification --dataset cifar10 --batch_size 1 --epoch 1

import argparse
from core.backbones import *
from core.utils.data_loader import load_data
#from core.utils.trainer import load_trainer  

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

ap = argparse.ArgumentParser()
ap.add_argument("--backbone", required=True, help="Choose backbone model")
ap.add_argument("--task", required=True, help="Choose task type")
ap.add_argument("--dataset", required=True, help="Choose dataset")
ap.add_argument("--batch_size", required=True, help="Set batch size")
ap.add_argument("--epoch", required=True, help="Set epoch")
ap.add_argument("--input_shape", required=False, help="Set input shape")

args = vars(ap.parse_args())
model_name = args["backbone"]
task_type = args["task"]
dataset = args["dataset"]
batch_size = int(args["batch_size"])
epoch = int(args["epoch"])
input_shape = list(args["input_shape"].split(","))
h = input_shape[0]
w = input_shape[1]
c = input_shape[2]

model_dict = {"Alexnet", "vgg8"}
task_name = {"classification", "detection"}
dataset_name = {"cifar10"}

#amc_method = getattr(backbones, model_name)


def main():
    assert model_name in model_dict, f'This repository don\'t support {model_name}' 
    assert task_type in task_name, f'{task_type} is wrong task type'
    assert dataset in dataset_name, f'This repository don\'t support {dataset}'
    assert batch_size > 0, 'Batch size must be lager than 1'
    assert epoch > 0, 'Epoch must be lager than 1'
    
    train_ds, val_ds = load_data(dataset, batch_size)
     
    model = load_trainer(model_name, task_type, input_shape=[h, w, c])
    print(input_shape)
    print(type(input_shape))

if __name__ == "__main__":
    main()