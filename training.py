"""
This is the main file used to start training.

Before starting, setup the following paths:

checkpoint_dir: Path where model checkpoints are saved
index_file: Path where csv file of the dataset is saved
model_path: Path where model after training to be saved
images_dir: Path where images are saved
results_path: Path where the results of model (loss, accuracy etc) are to be saved
history_model: Path where the history of model is to be saved
model_plot: Path where plot of model architecture to be saved
loss_plot: Path where plot of train/validation loss to be saved
accuracy_plot: Path where plot of train/validation accuracy to be saved
"""

import os
import time
import logging
import keras
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.utils import plot_model
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau
from preprocessing import resize_image
from model import resnet_model, cnn_model
from utils import convert_time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

matplotlib.use('agg')
start_time = time.time()

# Paths for dataset
checkpoint_dir = '/.../MuMu/checkpoints/genre-classifier.h5'
index_file = '/.../MuMu/working_files/train.csv'
model_path = '/.../MuMu/model/genre-classifier.h5'
images_dir = '/.../MuMu/Images/'
results_path = '/.../MuMu/results/'
history_model = '/.../MuMu/history.csv'
model_plot = '/.../MuMu/model/model.png'
loss_plot = '/.../MuMu/model/loss.png'
accuracy_plot = '/.../MuMu/model/accuracy.png'

# Training Parameters
BATCH_SIZE = 32
num_epochs = 300
nrows = 144
ncols = 144
learning_mode = 'variable' #fixed or #variable
learning_rate = 0.0000001
lr_min = 0.000000001
val_size = 0.2

# Customized class weights for imbalance classes
cw = {0: 1.0, 1: 1.2, 2: 0.82, 3: 0.70, 4: 1.1}

continue_training = False  # boolean

# Results directory setup
if not os.path.exists(results_path):
    os.makedirs(results_path)

# Read the CSV file using Pandas Dataframe
logger.info("Reading csv ...")
train = pd.read_csv(index_file)
genre_list = train.columns.tolist()
print("\ngenres for training are: {}\n".format(genre_list[1:]))

"""
tqdm is a progress bar just to see how much data is being processed. 
In the loop we are loading images from the folder Images according to 
Id in the csv file. 
After loading images we are converting them to arrays.
"""

logger.info("loading images...")

"""
Dataset contains images with different extensions, to load them the 
code is below
"""
train_image = []
for i in tqdm(range(train.shape[0])):
    image_path = images_dir + train['image_id'][i]
    if os.path.exists(image_path + '.jpg'):
        img_orig = image.load_img(image_path + '.jpg')
    if os.path.exists(image_path + '.png'):
        pass
    if os.path.exists(image_path + '.gif'):
        pass
    img = np.array(resize_image(img_orig, (nrows, ncols)))
    img = img / 255.0
    train_image.append(img)

logger.info("creating input dataset...")
# Now X contains all the data
X = np.array(train_image)

logger.info("images loaded....")

""" Load Genre(labels) of the movies according to the Id 
or the filename
"""

logger.info("creating labels...")

y = np.array(train.drop(['image_id'], axis=1))

"""
Create train/validation split of 80-20. 

X_train and X_val contain the images for train and val
y_train and y_val contain labels(i.e genres) for train and val
"""

logger.info("splitting input dataset...")

X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                  random_state=42,
                                                  test_size=val_size)

"""Checkpoint to save the best model during training with respect to 
validation accuracy if you want the save the best model w.r.t to training 
accuracy or loss just replace 'val_acc' with 'val_loss' or 'loss' or 'acc'.
CSVLogger is used to store the accuracy, loss values to csv.
ReduceLROnPlateau is used for varying learning rates during training"""

cb_checkpoint = ModelCheckpoint(checkpoint_dir,
                                monitor='val_accuracy',
                                verbose=1,
                                save_best_only=True,
                                save_weights_only=False,
                                mode='auto')

cb_CSVLogger = CSVLogger(history_model)

cb_EarlyStopping = EarlyStopping(monitor='val_loss')

cb_lr_factor = 0.1
cb_lr_patience = 5
cb_lr_cooldwon = 1

cb_ReduceLROnPlateau = ReduceLROnPlateau(monitor='val_loss',
                                         factor=cb_lr_factor,
                                         patience=cb_lr_patience,
                                         verbose=1,
                                         mode='auto',
                                         cooldown=cb_lr_cooldwon,
                                         min_lr=lr_min)

if learning_mode == 'variable':
    cb_list = [cb_checkpoint, cb_CSVLogger, cb_ReduceLROnPlateau]
elif learning_mode == 'fixed':
    cb_list = [cb_checkpoint, cb_CSVLogger]

logger.info("continuing training from last saved checkpoint...")
if continue_training == True:
    model = keras.models.load_model(model_path)
    model.evaluate(X_val, y_val)
elif continue_training == False:
    logger.info("loading cnn model...")
    model = cnn_model((nrows, ncols, 3), learning_rate, len(genre_list)-1)

logger.info("saving training configuration information ...")

# Text file containing training configuration is saved
text_file = open(results_path + "details.txt", "w")

text_file.write("results path: {}\ntrain .csv: {}\nepochs: {}\nlearning rate: {}"
                "\nbatch size: {}\nclasses: {} "
                "\nweights: {}\nimage size: {}\ntrain_test split: {}" .format(results_path, index_file,
                                                                              num_epochs, learning_rate,
                                                                              BATCH_SIZE, genre_list,
                                                                              cw, nrows, val_size))

if learning_mode == 'variable':
    text_file.write('\n\n\n\nminimum learning rate: {}\nfactor: {}\npatience: {}\ncooldown: {}'.format(lr_min,
                                                                                                       cb_lr_factor,
                                                                                                       cb_lr_patience,
                                                                                                       cb_lr_cooldwon))

text_file.close()

logger.info("training in progress ...")

history = model.fit(X_train, y_train,
                    batch_size=BATCH_SIZE,
                    validation_data=(X_val, y_val),
                    epochs=num_epochs,
                    callbacks=cb_list,
                    class_weight=cw)

logger.info("training completed ...")

model.save(model_path)

# plot_model saves the model architecture to a file
plot_model(model,
           to_file=model_plot,
           show_shapes=False,
           show_layer_names=True,
           rankdir='TB',
           expand_nested=False,dpi=96)

logger.info("model saved...")

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.savefig(accuracy_plot)
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig(loss_plot)
plt.show()

with open(results_path + "details.txt", "a") as txt_file:
    txt_file.write("\nexecution time: %s" % (convert_time(time.time() - start_time)))
txt_file.close()

print('--- Execution time: {} ---'.format(convert_time(time.time() - start_time)))

