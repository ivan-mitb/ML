# 2018-6-9 ivan: initial commit

import pandas as pd

# read the CSV without header
# (discard row 0 if it contains the header)
# fix the malformed row 4817100 by removing columns 0:14
df = pd.concat(
    (pd.read_csv('kddcup.data.txt', error_bad_lines=False, header=None, engine='c', memory_map=True),
    pd.read_csv('kddcup.data.txt', header=None, skiprows=4817100-1, nrows=1, engine='c', memory_map=True).iloc[:, 14:]),
    ignore_index=True, copy=False)
if df.iloc[0, 0] == 'duration':
    df = df[1:]

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

# the dataset is now ready, in DataFrame 'df'.
# you should first split it into train/test, before doing any other processing

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1], test_size=0.1, random_state=4129)

# the dataset is now ready, in DataFrames x_train and x_test
