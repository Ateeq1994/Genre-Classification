# Application of Neural Networks for Image-based Genre Identification

## Overview of the Project:
Cover art (such as music, movies and books) is an essential part of today’s life. Together with being attractive, it also varies with genre which we quickly grasp using visual ques such as color, art and illustrations. This genre-wise subdivision based on visual traits is also driven by consumer market and so images, artworks, advertisements or covers, all are designed to appeal to the po-tential buyers. As such the cover art contains certain characteristics relating to its category/genre. This research work proposes a methodology based on a multi-layer Convolutional Neural Net-work (CNN) that can classify the visual appearance of a music album cover, movie poster or book cover into multiple genres.

## Configurations
* Python 2.7
* Keras 2.1
* Further dependencies:
   - numpy v1.18.1
   - Pillow v7.0.0
   - pandas
   
## User Manual of the Project:

<strong> Directory structure of this repository </strong>

* `index_files` contains csv files used in training and inference.
* `model_files` contains the saved model (.h5 file).
* `src` contains the script for training, inference, dataset and model creation.

<strong> Training Data </strong>

* The dataset used for training and test purpose was taken from amazon_metadata_MuMu.
* MuMu is a Multimodal Music dataset with multi-label genre annotations and album metadata gathered from Amazon.com.
* MuMu dataset consists of 31k music albums classified into 250 genre classes.
* Movie poster dataset consists of 7k movie posters classified into 25 genre classes.  

* Reference of the dataset can be found here:
  `https://arxiv.org/abs/1707.04916`
* First of all, labels in the csv file of dataset were transformed into one-hot encoded labels using `data_transformation.py`.
* After transformation, training and test datasets with reduced number of classes are created using `class_filtering.py`. 
* After filtering, train.csv and test.csv file are prepared having following format:
`< images_ID, one-hot encoded labels >`

* Total size of the training images: 6,094 images
* 80/20 split is done for training.

<strong> Detailed description of the model </strong>

Convolutional Neural Networks (CNN) approach has been used for model architecture.

Model architecture configuration is as follows:

* Conv2D(16, kernel_size=(3, 3), activation='relu')
* MaxPooling2D(pool_size=(2, 2))
* Dropout(0.25) 
* Conv2D(32, kernel_size=(3, 3), activation='relu')
* MaxPooling2D(pool_size=(2, 2))
* Dropout(0.25)
* Conv2D(32, kernel_size=(3, 3), activation='relu')
* MaxPooling2D(pool_size=(2, 2))
* Dropout(0.25)
* Conv2D(64, kernel_size=(3, 3), activation='relu')
* MaxPooling2D(pool_size=(2, 2))
* Dropout(0.25)
* Flatten()
* Dense(64, activation='relu')
* Dropout(0.1)
* Dense(32, activation='relu')
* Dropout(0.1)
* Dense(6, activation='sigmoid')

Furthermore, for second model an existing pre-trained image classification model, ResNet50 is used as a feature detector.

Reference work for Resnet50 can be found here:

https://medium.com/@14prakash/understanding-and-implementing-architectures-of-resnet-and-resnext-for-state-of-the-art-image-cf51669e1624

Weights of Resnet50 are freezed and a new classification head is added.

* Loss-Function: Binary Crossentropy,
see https://keras.io/api/losses/probabilistic_losses/#binarycrossentropy-class
* Optimizer: Adam, see https://keras.io/api/optimizers/adam/

## A list of the contained files:
The `src` folder contains the following files:
* `data_transformation.py` This script transforms labels in original csv file into one-hot encoded labels.
* `class_filtering.py` This script filters the classes from original dataset and prepares csv file for training and inference.
* `preprocessing.py` This script contains preprocessing functions.
* `utils.py` This script contains helping functions.
* `model.py` This script mainly contains configuration for model architectures.
* `training.py` This is the main file used to start training.
* `inference.py` This is the main file used to perform inference.

## Contact information:
* Shayan Ahmed (shayan.ahmed@tu-ilmenau.de)
* Muhammad Ateeque Zaryab (muhammad-ateeque.zaryab@tu-ilmenau.de)
