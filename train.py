#!/usr/bin/python
import numpy
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.optimizers import RMSprop
from keras.utils import np_utils

out_classes=4
batch_size=128
num_digits=10

def create_dataset():
    inx=[]
    outx=[]
    for i in range(101,1024):
        inx.append(bin_encode(i))
        outx.append(out_encode(i))
    return numpy.array(inx),np_utils.to_categorical(numpy.array(outx),out_classes)



def out_encode(i):
    if i%15==0: return [3]
    elif i % 5  == 0: return [2]
    elif i % 3  == 0: return [1]
    else:             return [0]

def bin_encode(i):
    return [i >> d & 1 for d in range(num_digits)]

def fizz_buzz_pred(i, pred):
    return [str(i), "fizz", "buzz", "fizzbuzz"][pred.argmax()]

def fizz_buzz(i):
    if   i % 15 == 0: return "fizzbuzz"
    elif i % 5  == 0: return "buzz"
    elif i % 3  == 0: return "fizz"
    else:             return str(i)

inx,outx=create_dataset()

model=Sequential()

model.add(Dense(64,input_shape=(num_digits,)))
model.add(Activation('selu'))
model.add(Dense(256))
model.add(Activation('selu'))
model.add(Dense(4))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer=RMSprop())
model.fit(inx,outx,nb_epoch=1000,batch_size=batch_size)

model.save("model2")
errors=0
correct=0

for i in range(1,101):
    x = bin_encode(i)
    y = model.predict(numpy.array(x).reshape(-1,10))
    print(fizz_buzz_pred(i,y))
    if fizz_buzz_pred(i,y) == fizz_buzz(i):
        correct = correct + 1
    else:
        errors = errors + 1

print("Errors :" , errors, " Correct :", correct)
