import csv
import numpy as np
import pandas as pd

a = open('training_input.csv')
b = open('training_target.csv')

ap_VA = pd.read_csv("training_input.csv", skiprows=0)
bp_BA = pd.read_csv("training_target.csv", skiprows=0)

csv_reader_a = csv.reader(a)
csv_reader_b = csv.reader(b)

training_input = np.array([])
training_target = np.array([])


for line in csv_reader_a:
    training_input = np.append(training_input, line)

for line in csv_reader_b:
    training_target = np.append(training_target, line)

a.close()
b.close()

df = ap_VA
df.insert(8, "BRKa", bp_BA['BRKa'], True)
df.insert(9, "BRKb", bp_BA['BRKb'], True)
df.insert(10, "BRKc", bp_BA['BRKc'], True)
df.insert(11, "BRKd", bp_BA['BRKd'], True)

print(training_input, training_target)
print(df)

