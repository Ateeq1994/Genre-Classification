"""
This file transforms original csv file of dataset
into a file containing images name and labels (one-hot encoded).

Before starting inference, make sure the file paths are
correctly setup.
csv_orig_path: Path of the original csv
csv_one_hot_path: Path where the one hot encoded csv file to be saved
"""

import pandas as pd
import numpy as np
import time
from utils import convert_time

start_time = time.time()

# File paths
csv_orig_path = '/.../MuMu_dataset_multi-label.csv'
csv_one_hot_path = '/.../working_files/'
if not os.path.exists(csv_one_hot_path):
    os.makedirs(csv_one_hot_path)

# Read csv file using pandas dataframe
df = pd.read_csv(csv_orig_path, usecols=['amazon_id', 'genres'])
print("\nshape of csv is: {}".format(np.shape(df)))

# Removing duplicates
df.drop_duplicates(subset='amazon_id', keep='first', inplace=True)
print("\nnew shape: {}".format(np.shape(df)))

# One_hot encoding
one_hot = df['genres'].str.get_dummies(sep=',')
one_hot.insert(0, 'image_id', df['amazon_id'].tolist())

# Write to csv file
one_hot.to_csv(csv_one_hot_path+'MuMu_one_hot.csv', index=False)
print(one_hot)

print('--- Execution time: {} ---'.format(convert_time(time.time() - start_time)))
