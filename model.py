"""
This script contains the configuration of genre classification model architecture.
"""

import keras
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Model, Sequential
from keras.applications.resnet import ResNet50
from keras.layers.pooling import GlobalAveragePooling2D

# Neural Network definition
def cnn_model(input_shape, lr, num_classes):
    """
    Using keras Sequential model for image classification.
    This function configures the layers of the model and
    then compiles the model.
    :param lr: learning rate
    :param num_classes: number of output classes
    :param input_shape (width, height)
    :return compiled model (model.compile)
    """

    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation="relu", input_shape=(input_shape)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(num_classes, activation='sigmoid'))

    opt = keras.optimizers.Adam(learning_rate=lr)

    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model

# Transfer learning model
def resnet_model(input_shape):
    """"
    Transfer learning- Reuse existing pre-trained image classification model,
    only retraining the top layer of the network that determines the classes
    an image can belong to.

    Using a pre-trained ResNet50 model as the feature detector.
    It loads the ResNet50 model pre-trained on Imagenet without the top layer,
    freezes it´s weights, and adds a new classification head.
    :param input_shape (width, height)
    :return compiled model (model.compile)
    """

    base_model = ResNet50(include_top=False,
                          weights='imagenet',
                          input_tensor=None,
                          input_shape=input_shape,
                          pooling=None
                          )

    model = GlobalAveragePooling2D()(base_model.output)
    preds = Dense(7, activation='sigmoid', name='album_cover')(model)

    model = Model(inputs=base_model.input, outputs=preds)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()
    return model


