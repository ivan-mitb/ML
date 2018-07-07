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

def cat2ord(x_train, x_test, cols=['protocol_type','service','flag']):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()

    # manually encode the 3 string cols as ordinals
    for i in cols:
        x_train[i] = le.fit_transform(x_train[i])
        x_test[i] = le.transform(x_test[i])
        # d = x_train[i].unique()
        # d = pd.Series(d.size, index=d)
        # x_train[i] = d[x_train[i]].reset_index(drop=True)
        # x_test[i] = d[x_test[i]].reset_index(drop=True)

# generates train.dat, test.dat, cats.dat
def make_data():
    df = init_dataset()

    # the processed dataset is now in DataFrame 'df'.
    # we first split it into train/test, before doing any analysis

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, :-2], df.iloc[:, -2:], test_size=0.1, stratify=df.attack_type, random_state=4129)
    del df

    # convert categoricals to ordinal (in-place)
    cat2ord(x_train, x_test)

    # dummy-encode the 3 categoricals, place into cats_test/cats_test
    from sklearn.preprocessing import OneHotEncoder
    hot = OneHotEncoder(sparse=True)
    cats_train = hot.fit_transform(x_train.loc[:, ['flag','protocol_type','service']])
    cats_test = hot.transform(x_test.loc[:, ['flag','protocol_type','service']])

    # throw away the 3 categorical cols and 'num_outbound_cmds'
    cols = ['flag','protocol_type','service','num_outbound_cmds']
    x_train.drop(columns=cols, inplace=True)
    x_test.drop(columns=cols, inplace=True)

    # this gives us the sparse matrix cats_train, 84 cols.
    #
    # we use TruncSVD to reduce this into a smaller dense matrix that we join back to x_train.
    # if this cannot be done, we just extract cols 74, 8-10,2,33,34,58,13 from the sparse
    # and join to x_train. These are the cols shown by the decision tree with highest
    # importance towards finding the rare classes r2l and u2r.

    # Ady's work
    from sklearn.decomposition import TruncatedSVD
    tSVD = TruncatedSVD(n_components=10, random_state = 4129)
    catsvd_train = tSVD.fit_transform(cats_train)
    catsvd_test = tSVD.transform(cats_test)

    # convert y.attack/attack_type to numeric
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train.attack = le.fit_transform(y_train.attack)
    y_test.attack = le.transform(y_test.attack)
    y_train.attack_type = le.fit_transform(y_train.attack_type)
    y_test.attack_type = le.transform(y_test.attack_type)
    # convert to np.array
    y_train = y_train.values
    y_test = y_test.values

    save_object([cats_train, cats_test, catsvd_train, catsvd_test], 'cats.dat')
    save_object([x_train, y_train], 'train.dat')
    save_object([x_test, y_test], 'test.dat')

def load_train(fn='train.dat'):
    return load_object(fn)

# the dataset is now ready, in DataFrames x_train and x_test
