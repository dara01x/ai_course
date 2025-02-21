#gradient descent machine learning algorithm
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as mt
import numpy as np
import time

iris = datasets.load_iris()
x = iris.data
y = iris.target
w = [0]*4
b = 0
sum = 0
a = 0.00012
h = 0.00001


def getRes(_w0, _w1,_w2,_w3,_b):
    sum = 0
    for i in range(150):
        sum = sum + (y[i] - (x[i,0]*_w0 + x[i,1]*_w1 + x[i,2]*_w2+x[i,3]*_w3 + _b ))**2
    res = 1/150 * sum
    return res


xtrain, xtest, ytrain,ytest = train_test_split(x,y,test_size = 0.3)
res = 0

yh = x[112,0] * w[0] + x[112,1] * w[1] + x[112,2] * w[2] + x[112, 3] * w[3] +b
print("predected result befor training: ", yh)
print("acual result: ", y[112])
oldw = [3]*4
st = time.time()
while(abs(oldw[3] - w[3]) >= 0.000009):
    oldw = w.copy()
    w[0] = w[0] - (a *(getRes(w[0]+h, w[1], w[2], w[3],b) - getRes(w[0], w[1], w[2], w[3],b))/h)
    w[1] = w[1] - (a *(getRes(w[0], w[1]+h, w[2], w[3],b) - getRes(w[0], w[1], w[2], w[3],b))/h)
    w[2] = w[2] - (a *(getRes(w[0], w[1], w[2]+h, w[3],b) - getRes(w[0], w[1], w[2], w[3],b))/h)
    w[3] = w[3] - (a *(getRes(w[0], w[1], w[2], w[3]+h,b) - getRes(w[0], w[1], w[2], w[3],b))/h)
    b = b - (a*(getRes(w[0], w[1], w[2], w[3],b)))

yh = x[112,0] * w[0] + x[112,1] * w[1] + x[112,2] * w[2] + x[112, 3] * w[3] +b
print("predected result after training", yh)
print("b : ", b)
print("w: ", w)
print("training compleated in : " ,time.time() - st )