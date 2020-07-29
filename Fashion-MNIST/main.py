#!/usr/bin/python
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix
from keras.utils import np_utils

BATCH=32
EPOCHS=100
num_classes=10

def test_model(model,xtest,ytest):
    loss,acc=model.evaluate(xtest,ytest)
    yout=model.predict(xtest)
    yout=np.argmax(yout,axis=1)
    ytest=np.argmax(ytest,axis=1)
    #print(yout[0])
    return loss,ytest,yout

def run_models():
    fashion_mnist=keras.datasets.fashion_mnist
    (x,y),(xtest,ytest) = fashion_mnist.load_data()

    x=x.reshape((x.shape[0],28,28,1))
    xtest=xtest.reshape((xtest.shape[0],28,28,1))

    x=x.astype('float32')/255
    xtest=xtest.astype('float32')/255

    y=np_utils.to_categorical(y,num_classes)
    ytest=np_utils.to_categorical(ytest,num_classes)

    print("Testing CNN Model\n")
    model=load_model('models/model_cnn_adam.h5')
    #model.summary()
    loss,gt,pred=test_model(model,xtest,ytest)

    confusion=confusion_matrix(gt,pred)
    print(np.array(confusion))

    fp=open("convolution-neural-net.txt","w")
    fp.write('Loss on Test Data : {}\n'.format(loss))
    fp.write("Accuracy on Test Data : {}\n".format(np.mean(np.array(gt)==np.array(pred))))
    fp.write("gt_label,pred_label \n")
    for i in range(len(gt)):
        fp.write("{},{}\n".format(gt[i],pred[i]))
    fp.close()

    print("\n\nTesting MLP Model\n")
    model=load_model('models/model_mlp.h5')
    #model.summary()
    loss,gt,pred=test_model(model,xtest,ytest)

    confusion=confusion_matrix(gt,pred)
    print(np.array(confusion))

    fp=open("multi-layer-net.txt","w")
    fp.write('Loss on Test Data : {}\n'.format(loss))
    fp.write("Accuracy on Test Data : {}\n".format(np.mean(np.array(gt)==np.array(pred))))
    fp.write("gt_label,pred_label \n")
    for i in range(len(gt)):
        fp.write("{},{}\n".format(gt[i],pred[i]))
    fp.close()

def main():
    run_models()

main()
