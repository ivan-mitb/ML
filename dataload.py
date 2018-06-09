# 2018-6-9 ivan: initial commit

import pandas as pd

# read the CSV without header
# (discard row 0 if it contains the header)
df = pd.read_csv('kddcup.data.txt', nrows=5, header=None)
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

# the dataset is now ready, in DataFrame 'df'
