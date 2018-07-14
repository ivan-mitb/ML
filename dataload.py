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

# loads the corrected dataset, 311029 rows
def init_corrected(filename='corrected'):
    df = pd.read_csv(filename, header=None, engine='c', memory_map=True)

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

    # unknown attack_types are NaN; 18729 rows
    df.attack_type.value_counts(dropna=False)

    return df

# convert categoricals to ordinal (in-place)
# novel classes in 'corrected' are appended
def cat2ord(x_train, x_test, x_corr, cols=['protocol_type','service','flag']):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    # manually encode the 3 string cols as ordinals
    for i in cols:
        x_train[i] = le.fit_transform(x_train[i])
        x_test[i] = le.transform(x_test[i])
        new = set(x_corr[i]) - set(le.classes_)
        le.classes_ = np.concatenate((le.classes_, list(new)))
        x_corr[i] = le.transform(x_corr[i])
        # d = x_train[i].unique()
        # d = pd.Series(d.size, index=d)
        # x_train[i] = d[x_train[i]].reset_index(drop=True)
        # x_test[i] = d[x_test[i]].reset_index(drop=True)

# generates train.dat, test.dat, cats.dat
def make_data():
    df = init_dataset()
    df2 = init_corrected()

    # the processed dataset is now in DataFrame 'df'.
    # we first split it into train/test, before doing any analysis

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, :-2], df.iloc[:, -2:], test_size=0.1, stratify=df.attack_type, random_state=4129)
    x_corr, y_corr = df2.iloc[:, :-2], df2.iloc[:, -2:]
    del df, df2

    # convert categoricals to ordinal (in-place)
    cat2ord(x_train, x_test, x_corr)

    # dummy-encode the 3 categoricals, place into cats_test/cats_test
    from sklearn.preprocessing import OneHotEncoder
    hot = OneHotEncoder(sparse=True, handle_unknown='ignore')
    cats_train = hot.fit_transform(x_train.loc[:, ['flag','protocol_type','service']])
    cats_test = hot.transform(x_test.loc[:, ['flag','protocol_type','service']])
    cats_corr = hot.transform(x_corr.loc[:, ['flag','protocol_type','service']])

    # throw away the 3 categorical cols and 'num_outbound_cmds'
    cols = ['flag','protocol_type','service','num_outbound_cmds']
    x_train.drop(columns=cols, inplace=True)
    x_test.drop(columns=cols, inplace=True)
    x_corr.drop(columns=cols, inplace=True)

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
    catsvd_corr = tSVD.transform(cats_corr)

    if save_intermediates:
        save_object([cats_train, cats_test, catsvd_train, catsvd_test], 'cats.dat')
        save_object([x_train, y_train], 'train.dat')
        save_object([x_test, y_test], 'test.dat')

    #   FEATURE SCALING
    # we need to minmaxscale x_train to [0,1].

    from sklearn.preprocessing import MinMaxScaler

    mms = MinMaxScaler(copy=False)      # do it in-place
    x_train = mms.fit_transform(x_train)
    x_test = mms.transform(x_test)
    x_corr = mms.transform(x_corr)
    # catsvd_train/test these don't need scaling, they are binary dummy vars

    # join catsvd_train/test as well as the 9 important raw cols of cats_train/test
    # to x_train/test
    # this also converts x_train/test into np.array format (ie we lose column names)
    # now all columns are numeric.

    cols = [2,8,9,10,13,33,34,58,74]
    x_train = np.hstack([x_train, catsvd_train, cats_train[:, cols].toarray()])
    x_test = np.hstack([x_test, catsvd_test, cats_test[:, cols].toarray()])
    x_corr = np.hstack([x_corr, catsvd_corr, cats_corr[:, cols].toarray()])
    del cats_train, cats_test, cats_corr, catsvd_train, catsvd_test, catsvd_corr

    # READY.DAT - 56 cols
    # 56 cols: x_train cols 0-36, catsvd_train cols 37-46, cats_train cols 47-55
    save_object([x_train, x_test, y_train, y_test], 'ready.dat')
    save_object([x_corr, y_corr], 'corrected.dat')


# returns 5 princomps + 11 best cols of x_train/x_test
# generates REDUCE.DAT
def make_reduce(x_train, x_test, y_train, y_test, x_corr, y_corr):
    col_list = [0, 1, 2, 6, 10, 14, 17, 18, 27, 30, 31]
    ## Get top 5 PCA components
    from sklearn.decomposition import PCA
    pca = PCA(n_components = 5, random_state = 4129)
    pca_result = pca.fit_transform(x_train[:, :37])
    # join the 5 princomps with the 11 best columns
    x_train = np.hstack((pca_result, x_train[:, col_list], x_train[:, 37:]))
    x_test = np.hstack((pca.transform(x_test[:, :37]), x_test[:, col_list], x_test[:, 37:]))
    x_corr = np.hstack((pca.transform(x_corr[:, :37]), x_corr[:, col_list], x_corr[:, 37:]))
    # make REDUCE.DAT
    save_object([x_train, x_test, y_train, y_test], 'reduce.dat')
    save_object([x_corr, y_corr], 'corrected.dat')
