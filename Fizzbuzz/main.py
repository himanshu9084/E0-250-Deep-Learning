#!/usr/bin/python
import numpy
import sys
import os
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras.models import load_model

out_classes=4
batch_size=128
num_digits=10

def fixx(n,fw):
    t=0
    if(int(n)%3==0):
        #print('fizz',end='')
        t=3
    if(int(n)%5==0):
        #print('buzz',end='')
        if(t==3):
            t=15
        else:
            t=5
    if(t==0):
        #print(n,end='')
        fw.write(n)
    else:
        #print("\n",end='')
        if(t==3):
            fw.write('fizz\n')
        if(t==5):
            fw.write('buzz\n')
        if(t==15):
            fw.write('fizzbuzz\n')

def software1(file):
    fp=open(file,"r")
    fw=open("Software1.txt","w")
    for n in fp:
        fixx(n,fw)
    print('\nSoftware1.txt generated, logic based\n')
    fp.close()
    fw.close()

def bin_encode(i):
    return [i >> d & 1 for d in range(num_digits)]

def fizz_buzz_pred(i, pred):
    return [str(i), "fizz", "buzz", "fizzbuzz"][pred.argmax()]

def fizz_buzz(i):
    if   i % 15 == 0: return "fizzbuzz"
    elif i % 5  == 0: return "buzz"
    elif i % 3  == 0: return "fizz"
    else:             return str(i)

def outModel(file):
    model=load_model('model1')
    fp=open(file,"r")
    fw=open("Software2.txt","w")
    errors=0
    correct=0
    count=0
    for n in fp:
        i=int(n)
        count=count+1
        x=bin_encode(i)
        y = model.predict(numpy.array(x).reshape(-1,10))
        print(fizz_buzz_pred(i,y))
        fw.write(fizz_buzz_pred(i,y)+'\n')
        if fizz_buzz_pred(i,y) == fizz_buzz(i):
            correct = correct + 1
        else:
            errors = errors + 1
    fp.close()
    fw.close()
    print("Errors :" , errors, " Correct :", correct)
    print("Accuracy : ",(correct/count)*100,'%')
    print('\nSoftware2.txt generated, ML based')

def software2(file):
    if os.path.isfile('model1'):
        print('\nModel exists')
        outModel(file)
    else:
        print("\nModel doesn't exists")

def main():
    file=sys.argv[2]
    software1(file)
    software2(file)

main()
