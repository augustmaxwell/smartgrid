import numpy as np
import pandas as pd
from tensorflow import keras
from tkinter import *
import turtle
import time
from graphics import *
import socket
import struct

ap_VA = pd.read_csv("training_input-1.csv", skiprows=0)
bp_BA = pd.read_csv("training_target-1.csv", skiprows=0)

df = ap_VA
bp_BA.columns = ['BRKa', 'BRKb', 'BRKc', 'BRKd']
df.insert(8, "BRKa", bp_BA['BRKa'], True)
df.insert(9, "BRKb", bp_BA['BRKb'], True)
df.insert(10, "BRKc", bp_BA['BRKc'], True)
df.insert(11, "BRKd", bp_BA['BRKd'], True)

df.columns = ['VAANG', 'VAMAG', 'VBANG', 'VBMAG', 'VCANG', 'VCMAG', 'VDANG', 'VDMAG', 'BRKa', 'BRKb', 'BRKc', 'BRKd']

df['BRKs'] = df[df.columns[8:]].apply(
    lambda x: ''.join(x.dropna().astype(str)),
    axis=1
)
newdf = df.replace({'1110': '1111', '1101': '1111', '1011': '1111', '0111': '1111'})

one_hot_BRK = pd.get_dummies(newdf.BRKs).values

print(one_hot_BRK)

print(df.head())
# df.head().to_csv('Combined_Data.csv')
print(df.BRKa.unique())

model = keras.models.Sequential([
    keras.layers.Dense(500, input_shape=(8,), activation='softplus'),
    keras.layers.Flatten(),
    keras.layers.Dense(12, )]
)


model.compile(optimizer='Adam', loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

x = np.column_stack((newdf.VAANG.values, newdf.VAMAG.values, newdf.VBANG.values, newdf.VBMAG.values, newdf.VCANG.values,
                     newdf.VCMAG.values, newdf.VDANG.values, newdf.VDMAG.values))

np.random.RandomState(seed=400).shuffle(x)
np.random.RandomState(seed=400).shuffle(one_hot_BRK)
model.fit(x, one_hot_BRK, batch_size=24, epochs=15)

vail_df = newdf
vail_x = np.column_stack((newdf.VAANG.values, newdf.VAMAG.values, newdf.VBANG.values, newdf.VBMAG.values, newdf.VCANG.values,
                         newdf.VCMAG.values, newdf.VDANG.values, newdf.VDMAG.values))

vail_one_hot_BRK = pd.get_dummies(newdf.BRKs).values
print('Validation')
model.evaluate(vail_x, vail_one_hot_BRK)

ap_VA2 = pd.read_csv("testing_input-4.csv", skiprows=0)
bp_BA2 = pd.read_csv("testing_target-3.csv", skiprows=0)

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

newdf2 = df.replace({'1110': '1111', '1101': '1111', '1011': '1111', '0111': '1111'})
# newdf2.head().to_csv('Combined_Testing_Data.csv')


test_df = newdf
test_x = np.column_stack((newdf2.VAANG.values, newdf2.VAMAG.values, newdf2.VBANG.values, newdf2.VBMAG.values, newdf2.VCANG.values,
                          newdf2.VCMAG.values, newdf2.VDANG.values, newdf2.VDMAG.values))

test_one_hot_BRK = pd.get_dummies(newdf2.BRKs).values

print('Testing')
test_val = pd.DataFrame(model.evaluate(test_x, test_one_hot_BRK))
# test_val.to_csv('test_val.csv')

print("Prediction")
pred_1 = test_x
yhat = model.predict(pred_1)
yhat_1 = np.argmax(yhat, axis=1)

yhat_bin = pd.get_dummies(yhat_1).values
comp = []
predict = pd.DataFrame(yhat_1)
predict = predict.to_numpy()
for n in range(len(yhat_bin)):
    comp.append(np.array_equiv(yhat[n], test_one_hot_BRK[n]))
comp = pd.DataFrame(comp)
# comp.to_csv('Comparison.csv')

print('Prediction Ready')

win = GraphWin("My Window", 500, 500)
win.setBackground('white')
def graph(predict):
    rect = Rectangle(Point(150,350), Point(350,100))
    rect.setOutline('black')
    rect.setFill('white')
    rect.draw(win)

    if predict[0] == 15 or predict[0] == 14 or predict[0] == 13 or predict[0] == 11 or predict[0] == 7:
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

    if predict[0] == 9:
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
    if predict[0] == 10:
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
    if predict[0] == 12:
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
    if predict[0] == 5:
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
    if predict[0] == 3:
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
    if predict[0] == 6:
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
    if predict[0] == 1:
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
    if predict[0] == 2:
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
    if predict[0] == 4:
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
    if predict[0] == 8:
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
    if predict[0] == 0:
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

    # win.update()
    win.getMouse()
    win.close()
# RTDS = ('130.127.88.141', 5890)
# HOST = '130.127.88.141'
# PORT = 5890
# msg = b''
# msg_length = 4 * 8
#
# with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#     s.connect(RTDS)
#     i = 1
#     while (1):
#         data = 0
#         i = i + 1
#         data = s.recv(2048)
#         data = struct.unpack('>%sf' % (len(data) // 4), data)
#         buff = data[0:8]
#         yhat = model.predict(np.asarray([buff]))
#         yhat_1 = np.argmax(yhat, axis=1)
#         predict = yhat_1
predict = [5]
graph(predict)
