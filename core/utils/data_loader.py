import tensorflow as tf
from tensorflow.keras.datasets import cifar10


def load_data(dataset):
    if dataset == "cifar10":
        from tensorflow.keras.datasets import cifar10
        from tensorflow.keras.utils import to_categorical        
    
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        
        return x_train, y_train, x_test, y_test 