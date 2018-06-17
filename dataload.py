# 2018-6-9 ivan: initial commit
# 2018-6-17 put everything inside functions

import numpy as np
import pandas as pd

import pickle

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    obj = {}
    with open(filename, 'rb') as input:
        obj = pickle.load(input)
    return obj

# process the dataset from raw file to DataFrame
# returns df
def init_dataset(filename='kddcup.data.txt'):
    # read the CSV without header
    # (discard row 0 if it contains the header)
    # fix the malformed row 4817100 by removing columns 0:14
    df = pd.read_csv(filename, error_bad_lines=False, header=None, engine='c', memory_map=True)
    df1 = pd.read_csv(filename, header=None, skiprows=4817100-1, nrows=1, engine='c', memory_map=True).iloc[:, 14:]
    df1.columns = df.columns
    df = df.append(df1)
    if df.iloc[0, 0] == 'duration':
        df = df[1:]
    del(df1)
    df.reset_index(drop=True, inplace=True)

    # read in the headers (exclude first row)
    header = open('kddcup.names').readlines()[1:]
    header = [d.split(':')[0] for d in header]

    # set the column names on the DataFrame
    df.columns = header + ['attack']

    # remove trailing '.' in the attack labels
    df['attack'] = df['attack'].str.slice(0, -1)

    # add new column 'attack_type' containing
    #     dos, u2r, r2l, probe, normal
    attack_types = [d.split() for d in open('training_attack_types.txt').readlines()[:-1]]
    attack_types.append(['normal', 'normal'])
    attack_types = np.array(attack_types)
    attack_types = pd.DataFrame({ 'attack_type' : attack_types[:,1] },
        index=attack_types[:,0])
    df['attack_type'] = attack_types.loc[df.attack].attack_type.values
    return df

from sklearn.model_selection import train_test_split

def make_data():
    df = init_dataset()

    # the processed dataset is now in DataFrame 'df'.
    # we first split it into train/test, before doing any analysis

    x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, :-2], df.iloc[:, -2:], test_size=0.1, random_state=4129)

    save_object([x_train, y_train], 'train.dat')
    save_object([x_test, y_test], 'test.dat')

# the dataset is now ready, in DataFrames x_train and x_test
