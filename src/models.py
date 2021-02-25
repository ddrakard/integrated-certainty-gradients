import tensorflow.python.keras as keras


def greyscale_image_classifier() -> keras.Model:
    """ Crate a model to classify 28 x 28 pixel greyscale images. """
    number_of_classes = 10
    model = keras.Sequential()
    model.add(keras.Conv2D(32, kernel_size=(3, 3), activation='relu',
                           input_shape=(28, 28, 1)))
    model.add(keras.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.Dropout(0.25))
    model.add(keras.Flatten())
    model.add(keras.Dense(128, activation='relu'))
    model.add(keras.Dropout(0.5))
    model.add(keras.Dense(number_of_classes, activation='softmax'))
    return model


def value_confidence_image_classifier() -> keras.Model:
    """
        Crate a model to classify 28 x 28 pixel two channel (value and
        confidence) images.
    """
    number_of_classes = 10
    model = keras.Sequential()
    model.add(keras.Conv2D(
        32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 2)))
    model.add(keras.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.Dropout(0.25))
    model.add(keras.Flatten())
    model.add(keras.Dense(128, activation='relu'))
    model.add(keras.Dropout(0.5))
    model.add(keras.Dense(number_of_classes, activation='softmax'))
    return model
