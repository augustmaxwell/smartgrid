import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

train_df = pd.read_csv('training_input.csv')
target_df = pd.read_csv('training_target.csv')

ap_VA = pd.read_csv("training_input.csv", skiprows=0)
bp_BA = pd.read_csv("training_target.csv", skiprows=0)

df = ap_VA
df.insert(8, "BRKa", bp_BA['BRKa'], True)
df.insert(9, "BRKb", bp_BA['BRKb'], True)
df.insert(10, "BRKc", bp_BA['BRKc'], True)
df.insert(11, "BRKd", bp_BA['BRKd'], True)

df.columns = ['VAANG', 'VAMAG', 'VBANG', 'VBMAG', 'VCANG', 'VCMAG', 'VDANG', 'VDMAG', 'BRKa', 'BRKb', 'BRKc', 'BRKd']
# original = df
df['BRKs'] = df[df.columns[8:]].apply(
    lambda x: ''.join(x.dropna().astype(str)),
    axis=1
)
one_hot_BRK = pd.get_dummies(df.BRKs).values

print(one_hot_BRK)

print(df.head())
print(df.BRKa.unique())

model = keras.Sequential([
    keras.layers.Dense(500, input_shape=(8,), activation='softplus'),
    #keras.layers.Dropout(0.4),
    #keras.layers.Dense(250, activation='softplus'),
    keras.layers.Dense(16, activation='sigmoid')]
)

model.compile(optimizer='Adam', loss=keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy'])

#BinaryCrossentropy(from_logits=False),MeanSquaredError(),
x = np.column_stack((df.VAANG.values, df.VAMAG.values, df.VBANG.values, df.VBMAG.values, df.VCANG.values,
                     df.VCMAG.values, df.VDANG.values, df.VDMAG.values))
#y = np.column_stack((df.BRKa.values, df.BRKb.values, df.BRKc.values, df.BRKd.values))
np.random.RandomState(seed=400).shuffle(x)
np.random.RandomState(seed=400).shuffle(one_hot_BRK)
model.fit(x, one_hot_BRK, batch_size=24, epochs=10)

vail_df = df
vail_x = np.column_stack((df.VAANG.values, df.VAMAG.values, df.VBANG.values, df.VBMAG.values, df.VCANG.values,
                     df.VCMAG.values, df.VDANG.values, df.VDMAG.values))
vail_one_hot_BRK = pd.get_dummies(df.BRKs).values
print('Validation')
model.evaluate(vail_x, vail_one_hot_BRK)

ap_VA2 = pd.read_csv("testing_input.csv", skiprows=0)
bp_BA2 = pd.read_csv("testing_target.csv", skiprows=0)

bp_BA2.columns = ['BRKa', 'BRKb', 'BRKc', 'BRKd']

df2 = ap_VA2
df2.insert(8, "BRKa", bp_BA2['BRKa'], True)
df2.insert(9, "BRKb", bp_BA2['BRKb'], True)
df2.insert(10, "BRKc", bp_BA2['BRKc'], True)
df2.insert(11, "BRKd", bp_BA2['BRKd'], True)

df2.columns = ['VAANG', 'VAMAG', 'VBANG', 'VBMAG', 'VCANG', 'VCMAG', 'VDANG', 'VDMAG', 'BRKa', 'BRKb', 'BRKc', 'BRKd']
df2['BRKs'] = df2[df2.columns[8:]].apply(
    lambda x: ''.join(x.dropna().astype(str)),
    axis=1
)

one_hot_BRK = pd.get_dummies(df2.BRKs).values

test_df = df
test_x = np.column_stack((df2.VAANG.values, df2.VAMAG.values, df2.VBANG.values, df2.VBMAG.values, df2.VCANG.values,
                     df2.VCMAG.values, df2.VDANG.values, df2.VDMAG.values))
test_one_hot_BRK = pd.get_dummies(df2.BRKs).values
print('Testing')
model.evaluate(test_x, test_one_hot_BRK)
