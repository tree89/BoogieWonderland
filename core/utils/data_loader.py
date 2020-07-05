import tensorflow as tf


def load_cifar10():
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


def transform_dataset(x_train, y_train, x_test, y_test, batch_size):
    BATCH_SIZE = batch_size
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(10000).batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds


def load_data(dataset, batch_size):
    if dataset == "cifar10":
        x_train, y_train, x_test, y_test = load_cifar10()
        train_ds, val_ds = transform_dataset(x_train, y_train, x_test, y_test, batch_size)

    return train_ds, val_ds


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_cifar10()
    print(x_train.shape, y_train.shape)
    train_ds, val_ds = transform_dataset(x_train, y_train, x_test, y_test, 64)
    print(dir(train_ds))
    cnt = 0
    for obj in train_ds:
        cnt += 1
    print(cnt)
    for obj in train_ds.take(1):
        print(obj[0].shape[0])
