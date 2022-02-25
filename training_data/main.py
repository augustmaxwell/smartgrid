import csv
import numpy as np

a = open('training_input.csv')
b = open('training_target.csv')

csv_reader_a = csv.reader(a)
csv_reader_b = csv.reader(b)

training_input = np.array([])
training_target = np.array([])


for line in csv_reader_a:
    training_input = np.append(training_input,line)

for line in csv_reader_b:
    training_target = np.append(training_target,line)

a.close()
b.close()

print(training_input,training_target)
