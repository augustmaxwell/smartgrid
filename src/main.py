import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tkinter import *
import turtle
import time
from graphics import *

# train_df = pd.read_csv('training_input.csv')
# target_df = pd.read_csv('training_target.csv')
#
# ap_VA = pd.read_csv("training_input-1.csv", skiprows=0)
# bp_BA = pd.read_csv("training_target-1.csv", skiprows=0)
#
# df = ap_VA
# bp_BA.columns = ['BRKa', 'BRKb', 'BRKc', 'BRKd']
# df.insert(8, "BRKa", bp_BA['BRKa'], True)
# df.insert(9, "BRKb", bp_BA['BRKb'], True)
# df.insert(10, "BRKc", bp_BA['BRKc'], True)
# df.insert(11, "BRKd", bp_BA['BRKd'], True)
#
# df.columns = ['VAANG', 'VAMAG', 'VBANG', 'VBMAG', 'VCANG', 'VCMAG', 'VDANG', 'VDMAG', 'BRKa', 'BRKb', 'BRKc', 'BRKd']
# # original = df
# df['BRKs'] = df[df.columns[8:]].apply(
#     lambda x: ''.join(x.dropna().astype(str)),
#     axis=1
# )
# newdf = df.replace({'1110': '1111', '1101': '1111', '1011': '1111', '0111': '1111'})
# # print('DF BRKA info', df['BRKa'])
# # for i in df['BRKs']:
# #     if i == '1110':
# #         i = '1111'
# one_hot_BRK = pd.get_dummies(newdf.BRKs).values
#
# print(one_hot_BRK)
#
# print(df.head())
# df.head().to_csv('Combined_Data.csv')
# print(df.BRKa.unique())
#
# model = keras.models.Sequential([
#     keras.layers.Dense(500, input_shape=(8,), activation='softplus'),
#     #keras.layers.Dropout(0.4),
#     #keras.layers.Dense(250, activation='softplus'),
#     keras.layers.Flatten(),
#     keras.layers.Dense(12, )] #activation='sigmoid')]
# )
#
#
# model.compile(optimizer='Adam', loss=keras.losses.BinaryCrossentropy(from_logits=True),
#               metrics=['accuracy'])
#
# #BinaryCrossentropy(from_logits=False),MeanSquaredError(),
# x = np.column_stack((newdf.VAANG.values, newdf.VAMAG.values, newdf.VBANG.values, newdf.VBMAG.values, newdf.VCANG.values,
#                      newdf.VCMAG.values, newdf.VDANG.values, newdf.VDMAG.values))
# # x = np.column_stack((newdf.VAMAG.values, newdf.VBMAG.values, newdf.VCMAG.values, newdf.VDMAG.values))
# #y = np.column_stack((df.BRKa.values, df.BRKb.values, df.BRKc.values, df.BRKd.values))
# np.random.RandomState(seed=400).shuffle(x)
# np.random.RandomState(seed=400).shuffle(one_hot_BRK)
# model.fit(x, one_hot_BRK, batch_size=24, epochs=15)
# # print("one hot")
# # print(one_hot_BRK)
#
# vail_df = newdf
# vail_x = np.column_stack((newdf.VAANG.values, newdf.VAMAG.values, newdf.VBANG.values, newdf.VBMAG.values, newdf.VCANG.values,
#                          newdf.VCMAG.values, newdf.VDANG.values, newdf.VDMAG.values))
# # vail_x = np.column_stack((newdf.VAMAG.values, newdf.VBMAG.values, newdf.VCMAG.values, newdf.VDMAG.values))
# vail_one_hot_BRK = pd.get_dummies(newdf.BRKs).values
# print('Validation')
# model.evaluate(vail_x, vail_one_hot_BRK)
#
# ap_VA2 = pd.read_csv("testing_input-4.csv", skiprows=0)
# bp_BA2 = pd.read_csv("testing_target-3.csv", skiprows=0)
#
# bp_BA2.columns = ['BRKa', 'BRKb', 'BRKc', 'BRKd']
#
# df2 = ap_VA2
# df2.insert(8, "BRKa", bp_BA2['BRKa'], True)
# df2.insert(9, "BRKb", bp_BA2['BRKb'], True)
# df2.insert(10, "BRKc", bp_BA2['BRKc'], True)
# df2.insert(11, "BRKd", bp_BA2['BRKd'], True)
#
# df2.columns = ['VAANG', 'VAMAG', 'VBANG', 'VBMAG', 'VCANG', 'VCMAG', 'VDANG', 'VDMAG', 'BRKa', 'BRKb', 'BRKc', 'BRKd']
# df2['BRKs'] = df2[df2.columns[8:]].apply(
#     lambda x: ''.join(x.dropna().astype(str)),
#     axis=1
# )
# newdf2 = df.replace({'1110': '1111', '1101': '1111', '1011': '1111', '0111': '1111'})
# newdf2.head().to_csv('Combined_Testing_Data.csv')
# #one_hot_BRK = pd.get_dummies(df2.BRKs).values
#
# test_df = newdf
# test_x = np.column_stack((newdf2.VAANG.values, newdf2.VAMAG.values, newdf2.VBANG.values, newdf2.VBMAG.values, newdf2.VCANG.values,
#                           newdf2.VCMAG.values, newdf2.VDANG.values, newdf2.VDMAG.values))
# # test_x = np.column_stack((newdf2.VAMAG.values, newdf2.VBMAG.values, newdf2.VCMAG.values, newdf2.VDMAG.values))
# test_one_hot_BRK = pd.get_dummies(newdf2.BRKs).values
# # print(test_one_hot_BRK)
# # print(test_one_hot_BRK)
# # print(test_x)
# print('Testing')
# test_val = pd.DataFrame(model.evaluate(test_x, test_one_hot_BRK))
# test_val.to_csv('test_val.csv')
#
# print("Prediction")
# pred_1 = test_x
# yhat = model.predict(pred_1)
# yhat_1 = np.argmax(yhat, axis=1)
#
# yhat_bin = pd.get_dummies(yhat_1).values
# comp = []
# predict = pd.DataFrame(yhat_1)
# for n in range(len(yhat_bin)):
#     comp.append(np.array_equiv(yhat[n], test_one_hot_BRK[n]))
# comp = pd.DataFrame(comp)
# comp.to_csv('Comparison.csv')
# predict.to_csv('Predictions.csv')
# print('Prediction Ready')

# This variable is for testing, comment this out when actually running
predict = 8
def main(predict):
    win = GraphWin("My Window", 500, 500)
    win.setBackground('white')
    rect = Rectangle(Point(150,350), Point(350,100))
    rect.setOutline('black')
    rect.setFill('white')
    rect.draw(win)
    while True:
        # time.sleep(0.5)
        #\for i in range(len(predict)):

        # Place predict = serial read information here
        # This should update live information
        if predict == 15 or predict == 14 or predict == 13 or predict == 11 or predict == 7:
            BRKad = Rectangle(Point(240, 90), Point(260, 110))
            BRKad.setFill("green")
            BRKad.draw(win)

            BRKbd = Rectangle(Point(140, 250), Point(160, 270))
            BRKbd.setFill("green")
            BRKbd.draw(win)

            BRKcd = Rectangle(Point(280, 340), Point(300, 360))
            BRKcd.setFill("green")
            BRKcd.draw(win)

            BRKdd = Rectangle(Point(340, 215), Point(360, 235))
            BRKdd.setFill("green")
            BRKdd.draw(win)

            ln = Line(Point(150,225), Point(75,225))
            ln2 = Line(Point(350, 175), Point(425, 175))
            ln3 = Line(Point(350, 275), Point(425, 275))
            ln4 = Line(Point(250, 350), Point(250, 425))
            ln2.draw(win)
            ln4.draw(win)
            ln3.draw(win)
            ln.draw(win)

        if predict == 9:
            BRKad = Rectangle(Point(240, 90), Point(260, 110))
            BRKad.setFill("green")
            BRKad.draw(win)

            BRKbd = Rectangle(Point(140, 250), Point(160, 270))
            BRKbd.setFill("red")
            BRKbd.draw(win)

            BRKcd = Rectangle(Point(280, 340), Point(300, 360))
            BRKcd.setFill("red")
            BRKcd.draw(win)

            BRKdd = Rectangle(Point(340, 215), Point(360, 235))
            BRKdd.setFill("green")
            BRKdd.draw(win)

            ln = Line(Point(150, 225), Point(75, 225))
            ln2 = Line(Point(350, 175), Point(425, 175))
            ln3 = Line(Point(350, 275), Point(425, 275))
            ln4 = Line(Point(250, 350), Point(250, 425))
            ln2.draw(win)
            ln4.draw(win)
            ln3.draw(win)
            ln.draw(win)
        if predict == 10:
            BRKad = Rectangle(Point(240, 90), Point(260, 110))
            BRKad.setFill("green")
            BRKad.draw(win)

            BRKbd = Rectangle(Point(140, 250), Point(160, 270))
            BRKbd.setFill("red")
            BRKbd.draw(win)

            BRKcd = Rectangle(Point(280, 340), Point(300, 360))
            BRKcd.setFill("green")
            BRKcd.draw(win)

            BRKdd = Rectangle(Point(340, 215), Point(360, 235))
            BRKdd.setFill("red")
            BRKdd.draw(win)

            ln = Line(Point(150, 225), Point(75, 225))
            ln2 = Line(Point(350, 175), Point(425, 175))
            ln3 = Line(Point(350, 275), Point(425, 275))
            ln4 = Line(Point(250, 350), Point(250, 425))
            ln2.draw(win)
            ln4.draw(win)
            ln3.draw(win)
            ln.draw(win)
        if  predict == 12:
            BRKad = Rectangle(Point(240, 90), Point(260, 110))
            BRKad.setFill("green")
            BRKad.draw(win)

            BRKbd = Rectangle(Point(140, 250), Point(160, 270))
            BRKbd.setFill("green")
            BRKbd.draw(win)

            BRKcd = Rectangle(Point(280, 340), Point(300, 360))
            BRKcd.setFill("red")
            BRKcd.draw(win)

            BRKdd = Rectangle(Point(340, 215), Point(360, 235))
            BRKdd.setFill("red")
            BRKdd.draw(win)

            ln = Line(Point(150, 225), Point(75, 225))
            ln2 = Line(Point(350, 175), Point(425, 175))
            ln3 = Line(Point(350, 275), Point(425, 275))
            ln4 = Line(Point(250, 350), Point(250, 425))
            ln2.draw(win)
            ln4.draw(win)
            ln3.draw(win)
            ln.draw(win)
        if predict == 5:
            BRKad = Rectangle(Point(240, 90), Point(260, 110))
            BRKad.setFill("red")
            BRKad.draw(win)

            BRKbd = Rectangle(Point(140, 250), Point(160, 270))
            BRKbd.setFill("green")
            BRKbd.draw(win)

            BRKcd = Rectangle(Point(280, 340), Point(300, 360))
            BRKcd.setFill("red")
            BRKcd.draw(win)

            BRKdd = Rectangle(Point(340, 215), Point(360, 235))
            BRKdd.setFill("green")
            BRKdd.draw(win)

            ln = Line(Point(150, 225), Point(75, 225))
            ln2 = Line(Point(350, 175), Point(425, 175))
            ln3 = Line(Point(350, 275), Point(425, 275))
            ln4 = Line(Point(250, 350), Point(250, 425))
            ln2.draw(win)
            ln4.draw(win)
            ln3.draw(win)
            ln.draw(win)
        if predict == 3:
            BRKad = Rectangle(Point(240, 90), Point(260, 110))
            BRKad.setFill("red")
            BRKad.draw(win)

            BRKbd = Rectangle(Point(140, 250), Point(160, 270))
            BRKbd.setFill("red")
            BRKbd.draw(win)

            BRKcd = Rectangle(Point(280, 340), Point(300, 360))
            BRKcd.setFill("green")
            BRKcd.draw(win)

            BRKdd = Rectangle(Point(340, 215), Point(360, 235))
            BRKdd.setFill("green")
            BRKdd.draw(win)

            ln = Line(Point(150, 225), Point(75, 225))
            ln2 = Line(Point(350, 175), Point(425, 175))
            ln3 = Line(Point(350, 275), Point(425, 275))
            ln4 = Line(Point(250, 350), Point(250, 425))
            ln2.draw(win)
            ln4.draw(win)
            ln3.draw(win)
            ln.draw(win)
        if predict == 6:
            BRKad = Rectangle(Point(240, 90), Point(260, 110))
            BRKad.setFill("red")
            BRKad.draw(win)

            BRKbd = Rectangle(Point(140, 250), Point(160, 270))
            BRKbd.setFill("green")
            BRKbd.draw(win)

            BRKcd = Rectangle(Point(280, 340), Point(300, 360))
            BRKcd.setFill("green")
            BRKcd.draw(win)

            BRKdd = Rectangle(Point(340, 215), Point(360, 235))
            BRKdd.setFill("red")
            BRKdd.draw(win)

            ln = Line(Point(150, 225), Point(75, 225))
            ln2 = Line(Point(350, 175), Point(425, 175))
            ln3 = Line(Point(350, 275), Point(425, 275))
            ln4 = Line(Point(250, 350), Point(250, 425))
            ln2.draw(win)
            ln4.draw(win)
            ln3.draw(win)
            ln.draw(win)
        if predict == 1:
            BRKad = Rectangle(Point(240, 90), Point(260, 110))
            BRKad.setFill("red")
            BRKad.draw(win)

            BRKbd = Rectangle(Point(140, 250), Point(160, 270))
            BRKbd.setFill("red")
            BRKbd.draw(win)

            BRKcd = Rectangle(Point(280, 340), Point(300, 360))
            BRKcd.setFill("red")
            BRKcd.draw(win)

            BRKdd = Rectangle(Point(340, 215), Point(360, 235))
            BRKdd.setFill("green")
            BRKdd.draw(win)

            ln = Line(Point(150, 225), Point(75, 225))
            ln2 = Line(Point(350, 175), Point(425, 175))
            ln3 = Line(Point(350, 275), Point(425, 275))
            ln4 = Line(Point(250, 350), Point(250, 425))
            ln2.draw(win)
            ln4.draw(win)
            ln3.draw(win)
            ln.draw(win)
        if predict == 2:
            BRKad = Rectangle(Point(240, 90), Point(260, 110))
            BRKad.setFill("red")
            BRKad.draw(win)

            BRKbd = Rectangle(Point(140, 250), Point(160, 270))
            BRKbd.setFill("red")
            BRKbd.draw(win)

            BRKcd = Rectangle(Point(280, 340), Point(300, 360))
            BRKcd.setFill("green")
            BRKcd.draw(win)

            BRKdd = Rectangle(Point(340, 215), Point(360, 235))
            BRKdd.setFill("red")
            BRKdd.draw(win)

            ln = Line(Point(150, 225), Point(75, 225))
            ln2 = Line(Point(350, 175), Point(425, 175))
            ln3 = Line(Point(350, 275), Point(425, 275))
            ln4 = Line(Point(250, 350), Point(250, 425))
            ln2.draw(win)
            ln4.draw(win)
            ln3.draw(win)
            ln.draw(win)
        if predict == 4:
            BRKad = Rectangle(Point(240, 90), Point(260, 110))
            BRKad.setFill("red")
            BRKad.draw(win)

            BRKbd = Rectangle(Point(140, 250), Point(160, 270))
            BRKbd.setFill("green")
            BRKbd.draw(win)

            BRKcd = Rectangle(Point(280, 340), Point(300, 360))
            BRKcd.setFill("red")
            BRKcd.draw(win)

            BRKdd = Rectangle(Point(340, 215), Point(360, 235))
            BRKdd.setFill("red")
            BRKdd.draw(win)

            ln = Line(Point(150, 225), Point(75, 225))
            ln2 = Line(Point(350, 175), Point(425, 175))
            ln3 = Line(Point(350, 275), Point(425, 275))
            ln4 = Line(Point(250, 350), Point(250, 425))
            ln2.draw(win)
            ln4.draw(win)
            ln3.draw(win)
            ln.draw(win)
        if predict == 8:
            BRKad = Rectangle(Point(240, 90), Point(260, 110))
            BRKad.setFill("green")
            BRKad.draw(win)

            BRKbd = Rectangle(Point(140, 250), Point(160, 270))
            BRKbd.setFill("red")
            BRKbd.draw(win)

            BRKcd = Rectangle(Point(280, 340), Point(300, 360))
            BRKcd.setFill("red")
            BRKcd.draw(win)

            BRKdd = Rectangle(Point(340, 215), Point(360, 235))
            BRKdd.setFill("red")
            BRKdd.draw(win)

            ln = Line(Point(150, 225), Point(75, 225))
            ln2 = Line(Point(350, 175), Point(425, 175))
            ln3 = Line(Point(350, 275), Point(425, 275))
            ln4 = Line(Point(250, 350), Point(250, 425))
            ln2.draw(win)
            ln4.draw(win)
            ln3.draw(win)
            ln.draw(win)
        if predict == 0:
            BRKad = Rectangle(Point(240, 90), Point(260, 110))
            BRKad.setFill("red")
            BRKad.draw(win)

            BRKbd = Rectangle(Point(140, 250), Point(160, 270))
            BRKbd.setFill("red")
            BRKbd.draw(win)

            BRKcd = Rectangle(Point(280, 340), Point(300, 360))
            BRKcd.setFill("red")
            BRKcd.draw(win)

            BRKdd = Rectangle(Point(340, 215), Point(360, 235))
            BRKdd.setFill("red")
            BRKdd.draw(win)

            ln = Line(Point(150, 225), Point(75, 225))
            ln2 = Line(Point(350, 175), Point(425, 175))
            ln3 = Line(Point(350, 275), Point(425, 275))
            ln4 = Line(Point(250, 350), Point(250, 425))
            ln2.draw(win)
            ln4.draw(win)
            ln3.draw(win)
            ln.draw(win)

        win.getMouse()
        win.close()
# predictions = [15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0]
# for predict in predictions:
main()
