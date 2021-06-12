"""
This script filters the classes from original dataset and prepares csv file for training and inference.
It also generates heatmap/data_distribution for the datasets.
"""

import os
import time
import tqdm
import heapq
import itertools
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from utils import convert_time

matplotlib.use('agg')
start_time = time.time()

# Paths for dataset
index_file = '/.../MuMu/working_files/MuMu_one_hot.csv'
images_path = '/.../MuMu/Images/'
destination_folder = '/.../MuMu/working_files/MuMu_one_hot_filtered.csv'
pred_csv = '/.../MuMu/working_files/predict_images.csv'
pred_csv_balanced = '/.../MuMu/working_files/predict_images_balanced.csv'
image_per_genre_graph = '/.../MuMu/model/image_per_genre.png'
working_files_path = '/.../MuMu/working_files/'
dataset_plot_path = '/.../MuMu/dataset_plots/'
heatmap_plot = '/.../MuMu/dataset_plots/data_distribution.png'
image_per_genre_train = '/.../MuMu/dataset_plots/train.png'
image_per_genre_predict = '/.../MuMu/dataset_plots/predict.png'

if not os.path.exists(dataset_plot_path):
    os.makedirs(dataset_plot_path)

if not os.path.exists(working_files_path):
    os.makedirs(working_files_path)

num_images = 1000
num_pred_img = 200

# Variable for MuMu data
#class_list = ['image_id', 'Adult Alternative', 'Dance & Electronic', 'Country', 'Indie & Lo-Fi', 'Rap & Hip-Hop']

def extract_df(path_to_csv, predict_list, path_to_predcsv, class_list):
    """
    Function to extract prediction data.
    :param path_to_csv:
    :param predict_list:
    :param path_to_predcsv:
    :param class_list:
    :return: prediction datatset
    """

    df = pd.read_csv(path_to_csv, index_col=False, usecols=class_list)
    print('predict list length: {}'.format(len(predict_list)))
    df_new = df.loc[df[df.columns[0]].isin(predict_list)]
    df_new.to_csv(path_to_predcsv, index=False)
    print('shape of prediction images dataframe is: {}'.format(np.shape(df_new)))
    print('csv for test images has been created.')

def class_filter(class_list, path_to_csv, destination_folder, num_img):
    """
    Function to keep only those samples which belong to either of the
    classes and limit around 1000 per class, also use remaining for
    prediction.
    :param class_list:
    :param path_to_csv:
    :param num_img:
    :param destination_folder:
    :return: filtered datatset
    """

    predict_list = []
    predict_indices = []
    indices_drop = []
    empty_list = [0 for i in range(len(class_list))]
    dic = dict(zip(class_list[1:], empty_list))
    df = pd.read_csv(path_to_csv, usecols=class_list, index_col=False)
    print('\ninput csv shape: {}\n'.format(np.shape(df)))
    print(class_list)

    for index, row in tqdm.tqdm(df.iterrows()):
        count = 0
        cnnt = 0

        for class_name in class_list[1:]:
            if row[class_name] == 1:
                if dic[class_name] < num_img and cnnt == 0:
                    cnnt += 1
                    dic[class_name] = dic[class_name] + 1
                else:
                    predict_indices.append(index)
                    predict_list.append(row[0])

        if cnnt == 0:
            indices_drop.append(index)
    from collections import OrderedDict
    predict_list = list(OrderedDict.fromkeys(predict_list))
    print('length of predict_indices is: {}'.format(len(predict_indices)))
    print('length of predict_list: {}'.format(len(predict_list)))
    print('indices_drop length: {}'.format(len(indices_drop)))
    print('shape of dataframe before drop is: {}'.format(np.shape(df)))
    df = df.drop(indices_drop)
    print('shape of dataframe after drop is: {}'.format(np.shape(df)))
    df.to_csv(destination_folder, index=False)

    for class_name in class_list[1:]:
        total = df[class_name].sum()
        print('number of images {} are {}'.format(class_name, total))
    print('dictionary count: {}'.format(dic))
    return df, predict_list

def total_per_ncat(index_file, top_n):
    """
    Function to calculate total items per category (genres)
    :param index_file:
    :param top_n:
    :return: top_classes, top_classes_count, classes
    """

    df = pd.read_csv(index_file)
    classes = np.array(df.columns[1:])
    count = []
    top_classes = []
    top_classes_count = []
    for class_name in classes:
        total = df[class_name].sum()
    count.append(total)
    ind = heapq.nlargest(top_n, range(len(count)), count.__getitem__)
    for i in ind:
        top_classes.append(classes[i])
        top_classes_count.append(count[i])
    return top_classes, top_classes_count, classes

def list2pairs(l):
    """
    Supporting function for viz
    """
    pairs = list(itertools.combinations(l, 2))
    for i in l:
        pairs.append([i,i])
    return pairs

def viz(index_file, genre_list, plot_save_destination):
    """
    Function to generate heatmap/data distribution for datasets.
    :param index_file:
    :param genre_list:
    :param plot_save_destination
    :return: heatmap
    """
    df = pd.read_csv(index_file, usecols=genre_list)
    df["sum"] = df.sum(axis=1)
    print('vizzzzzzz')

    for index, row in tqdm.tqdm(df.iterrows()):
        if df['sum'][index] == 0:
            df = df.drop(index)
    df = df.drop(columns=['sum'])

    df['genres'] = (df.stack()
                    .reset_index(name='val')
                    .query('val == 1')
                    .groupby('level_0')['level_1']
                    .apply(list))
    allPairs = []

    for index, row in tqdm.tqdm(df.iterrows()):
        allPairs.extend(list2pairs(df['genres'][index]))
    df = df.drop(columns=['genres'])
    genre_pairs = list2pairs(genre_list)
    unique_genre_pair = np.unique(genre_pairs)
    visGrid = np.zeros((len(unique_genre_pair), len(unique_genre_pair))).astype(int)
    for p in allPairs:
        visGrid[np.argwhere(unique_genre_pair == p[0]), np.argwhere(unique_genre_pair == p[1])] += 1
        if p[1] != p[0]:
            visGrid[np.argwhere(unique_genre_pair == p[1]), np.argwhere(unique_genre_pair == p[0])] += 1
    myArray = visGrid
    myInt = 31471
    newArray = (myArray / myInt) * 100
    newArray = (newArray).astype(int)
    cmap = sns.cm.rocket_r
    fig, ax = plt.subplots(figsize=(44, 44))
    sns.set(font_scale=4.0)
    ax = sns.heatmap(newArray,
                     xticklabels=df.columns.tolist(),
                     yticklabels=df.columns.tolist(),
                     annot=True, fmt='',
                     linewidths=0.5,
                     annot_kws={"size": 40},
                     cmap=cmap, )
    for t in ax.texts: t.set_text(t.get_text() + " %")

    plt.xticks(rotation=45)
    fig = ax.get_figure()
    plt.title('data_distribution')
    fig.savefig(plot_save_destination, bbox_inches='tight')
    print('plot saved to: {}'.format(heatmap_plot))

# Read csv and make a list of genres
df = pd.read_csv(index_file, index_col=False)
df = df.drop(df.columns[0], axis=1)
genre_list = df.columns.tolist()
print('genres are: {}'.format(genre_list))
print("shape of dataframe is: {}".format(np.shape(df)))

# Plot number of images per genre
image_per_genre = df.sum(axis=0)
print('image per genre = {}'.format(image_per_genre))
plt.figure()
plt.bar(np.arange(len(genre_list)), image_per_genre)
plt.xticks(np.arange(len(genre_list)), genre_list, rotation='vertical')
plt.xlabel('genre list')
plt.ylabel('number of images')
#plt.savefig(image_per_genre_graph, dpi=96, bbox_inches='tight')
plt.show()

# Sort classes and save new csv
df_new, predict_list = class_filter(class_list, index_file, destination_folder, num_images)
df2 = pd.read_csv(destination_folder, index_col=False)
df2 = df2.drop(df2.columns[0], axis=1)
genre_list2 = df2.columns.tolist()
print("shape of dataframe is: {}".format(np.shape(df2)))

# plotting number of images per genre
image_per_genre2 = df2.sum(axis=0)
print('image per genre = {}'.format(image_per_genre2))
plt.figure()
plt.bar(np.arange(len(genre_list2)), image_per_genre2)
plt.xticks(np.arange(len(genre_list2)), genre_list2, rotation='vertical')
plt.xlabel('genre list')
plt.ylabel('number of images')
plt.savefig(image_per_genre_train, dpi=96, bbox_inches='tight')
plt.show()

extract_df(index_file, predict_list, pred_csv, class_list)
class_filter(class_list, pred_csv, pred_csv_balanced, num_pred_img)
top_n_classes, classes = total_per_ncat(index_file, 20)
viz(index_file, top_n_classes, heatmap_plot)

print('--- Execution time: {} ---'.format(convert_time(time.time() - start_time)))

