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

vals= df.values

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
    if (add <= 3):
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
