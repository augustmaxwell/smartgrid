import csv
import numpy as np
import pandas as pd

ap_VA = pd.read_csv("training_input.csv", skiprows=0)
bp_BA = pd.read_csv("training_target.csv", skiprows=0)

df = ap_VA
df.insert(8, "BRKa", bp_BA['BRKa'], True)
df.insert(9, "BRKb", bp_BA['BRKb'], True)
df.insert(10, "BRKc", bp_BA['BRKc'], True)
df.insert(11, "BRKd", bp_BA['BRKd'], True)

vals = df.values

for i in vals:
    brka = i[8]
    brkb = i[9]
    brkc = i[10]
    brkd = i[11]
    #print("brka: %f" % (brka))
    #print("brkb: %f" % (brkb))
    #print("brkc: %f" % (brkc))
    #print("brkd: %f" % (brkd))

    #Each  individual case (number of breakers on)
    add = brka + brkb + brkc + brkd
    # 3+ Case
    if (add >= 3):
        print("This is the 3+ case")
    # 2 Case
    if (add == 2):
        print("This is the 2 case")
    # 1 Case
    if (add == 1):
        print("This is the 1 case")
    # 0 Case
    if (add == 0):
        print("This is the 0 case")

#putting data into np.array form
trn_input = open('training_input.csv', 'rb')
input_data = np.loadtxt('training_input.csv', delimiter=",", skiprows=1)
#print(input_data)
#print(input_data.shape)

trn_target = open('training_target.csv', 'rb')
target_data = np.loadtxt('training_target.csv', delimiter=",", skiprows=1)
#print(target_data)
#print(target_data.shape)

#assigning weights, these are just random apparently
weights = np.array([[0.1], [0.2], [0.1], [0.2], [0.1], [0.2], [0.1], [0.2]])
#print(weights)

#add bias, also random.... maybe?
bias = 0.3

#activation function
def sigmoid_func(x):
    return 1/(1+np.exp(-x))


#derivative of sigmoid function
def der(x):
    return sigmoid_func(x)*(1-sigmoid_func(x))

#updateing the weights
for epochs in range(10000):
    input_arr = input_data

    weighted_sum = np.dot(input_arr, weights) + bias
    first_output = sigmoid_func(weighted_sum)

    error = first_output - target_data
    total_error = np.square(np.subtract(first_output, target_data)).mean

    first_der = error
    second_der = der(first_output)
    derivative = first_der * second_der

    t_input = input_data.T
    final_der = np.dot(t_input, derivative)

    #update weights
    weights = weights - 0.05 * final_der

    #update bias
    for i in derivative:
        bias = bias - 0.05 * i

print(weights)
print(bias)

#predictions
pred = input_data[478]
results = np.dot(pred, weights) + bias
res = sigmoid_func(results)
print(res)
