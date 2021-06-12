"""
This is the main file used to perform inference.

Before starting inference, make sure the file paths are
correctly setup.

model_path: Path where the saved model is located
index_file: Path where the index file of test data is saved
predict_images: Path where the test images are located
cm_plot: Path where confusion matrix plot to be saved
classification_report_path: Path where Precision, Recall and F1 score to be saved
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from preprocessing import resize_image
from keras.preprocessing import image
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from utils import convert_time

matplotlib.use('agg')
cwd = os.getcwd()
print("current  directory:", cwd)

start_time = time.time()
con = os.listdir(cwd)
print(con)

# File paths
model_path = '/.../model/genre-classifier.h5'
index_file = '/.../working_files/predict_images.csv'
predict_images = '/.../predict_images/'
cm_plot = '/.../confusion_matrix.png'
classification_report_path = '/.../classification_report_all.csv'


# variables
nrows = 144  # image size
ncols = 144  # image size
y_true = []  # will contain true labels
y_pred = []  # will contain predicted labels

# This function prints and plots the confusion matrix
def plot_confusion_matrix(cm,
                          classes,
                          normalize=True,
                          title="Confusion Matrix",
                          cmap=plt.cm.Blues):
    cm[np.isnan(cm)] = 0
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Prediction label')

# Read csv file
df = pd.read_csv(index_file, index_col=False)
df["sum"] = df.sum(axis=1)

for index, row in tqdm(df.iterrows()):
    if row['sum'] == 0:
        df = df.drop(index)
df = df.drop(columns="sum")
classes = np.array(df.columns[1:])
print('\nclasses is: {}'.format(classes))

# loading the saved model
model = load_model(model_path)

# Start prediction after loading images
print("Starting predictions ...")
classes = classes.tolist()
ctr = 0
for index, row in tqdm(df.iterrows()):
    ctr += 1
    print(df.columns[1])
    image_path = images_dir + row[df.columns[0]]
    print("image is: {}".format(image_path))
    if os.path.exists(image_path + '.jpg'):
        image_load = image.load_img(image_path + '.jpg')
    if os.path.exists(image_path + '.png'):
        image_load = image.load_img(image_path + '.png')
    if os.path.exists(image_path + '.gif'):
        image_load = image.load_img(image_path + '.gif')

    img = np.array(resize_image(image_load, (nrows, ncols)))
    img = img / 255.0

    # prediction
    proba = model.predict(img.reshape(1, nrows, ncols, 3))
    print('\n\nproba is: {}'.format(proba[0]))
    top_3 = np.argsort(proba[0])[:-4:-1]
    genre_max_ind = np.argmax(proba)
    print('\nindex of maximum probability: {}'.format(genre_max_ind))
    correct_true_lab = []
    correct_pred_lab = []

    cnt = 0
    for class_name in classes:
        if row[class_name] == 1:
            correct_true_lab.append(classes.index(class_name))
    print('top_3 is: {}'.format((top_3)))
    print('correct true labels: {}'.format(correct_true_lab))

    check = any(item in top_3 for item in correct_true_lab)
    if check is True:
        yt = list(set(top_3) & set(correct_true_lab))
        y_true.extend(yt)
        y_pred.extend(yt)
    else:
        y_true.append(correct_true_lab[0])
        y_pred.append(genre_max_ind)

# To convert labels in y_true and y_pred to label indices.
for class_ind in range(0, len(classes)):
    for n, i in enumerate(y_true):
        if i == classes[class_ind]:
            y_true[n] = int(class_ind)

for class_ind in range(0, len(classes)):
    for n, i in enumerate(y_pred):
        if i == classes[class_ind]:
            y_pred[n] = int(class_ind)

# plotting confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=None)
plot_confusion_matrix(cm, classes, title='Confusion Matrix')
plt.savefig(cm_plot, bbox_inches='tight')
plt.show()

""" 
This section compares true labels(groundtruth) with the predictions by the 
model and prints the comparison result. Moreover, it calculates the accuracy as 
well. 
"""

counter = 0
for pred, gt in zip(y_pred, y_true):
    text = "Groundtruth: {}\tPrediction: {}".format(gt, pred)
    if pred == gt:
        text += "\tTRUE"
        counter += 1
    else:
        text += "\tFALSE"

print("\n{} von {} True. Accuracy: {:.2%}.".format(counter, len(y_pred), float(counter) / len(y_pred)))

# Precision, Recall and F1 score calculation
report_dict = classification_report(y_true, y_pred, output_dict = True)
report_df = pd.DataFrame(report_dict)
report_df.to_csv(classification_report_path)

print('--- Execution time: {} ---'.format(convert_time(time.time() - start_time)))

