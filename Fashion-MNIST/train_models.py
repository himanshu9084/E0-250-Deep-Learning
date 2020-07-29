#!/usr/bin/python
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout,Flatten,GaussianNoise
from keras.optimizers import RMSprop,SGD,Adam
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras


BATCH=32
EPOCHS=50
num_classes=10
num_epochs = np.arange(0, EPOCHS)

def build_CNN():
    model=Sequential()
    model.add(Conv2D(32,(3,3),padding='same',input_shape=(28,28,1)))
    model.add(GaussianNoise(0.1))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32,(3,3),padding='same'))
    model.add(GaussianNoise(0.1))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64,(3,3),padding='same'))
    model.add(Activation('selu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64,(3,3),padding='same'))
    model.add(Activation('selu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128,(3,3),padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128,(3,3),padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(10))
    model.add(Activation('softmax'))
    opt=Adam(lr=0.001,beta_1=0.9,beta_2=0.999)
    model.compile(loss='categorical_crossentropy',optimizer=opt,
    metrics=['accuracy'])
    return model

def build_MLP():
    model=Sequential()
    model.add(Dense(512,activation='relu',input_shape=(28,28,1)))
    model.add(Dropout(0.1))
    model.add(Dense(512,activation='selu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(num_classes,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer=RMSprop(),
    metrics=['accuracy'])
    return model

def plotting(history,name):
    plt.style.use('ggplot')
    plt.figure(figsize=(12, 8))
    plt.plot(num_epochs, history.history['loss'], label='train_loss', c='red')
    plt.title('Training Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.legend()
    plt.savefig(name)

def load_data():
    fashion_mnist=keras.datasets.fashion_mnist
    (x,y),(xtest,ytest) = fashion_mnist.load_data()
    print('x shape',x.shape[0])

    x=x.reshape((x.shape[0],28,28,1))
    xtest=xtest.reshape((xtest.shape[0],28,28,1))

    x=x.astype('float32')/255
    xtest=xtest.astype('float32')/255

    y=np_utils.to_categorical(y,num_classes)
    ytest=np_utils.to_categorical(ytest,num_classes)


    augment=keras.preprocessing.image.ImageDataGenerator(rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.09,height_shift_range=0.09)
    augment.fit(x)

    print("Building CNN Model \n")
    model_cnn=build_CNN()
    history=model_cnn.fit_generator(augment.flow(x,y,batch_size=BATCH),
    epochs=EPOCHS,validation_data=(xtest,ytest))
    model_cnn.save("models/model_cnn_1.h5")
    plotting(history,"models/cnn_adam_graph_1.png")

    augment=keras.preprocessing.image.ImageDataGenerator()
    augment.fit(x)
    print("Building MLP Model\n")
    model_mlp=build_MLP()
    history=model_mlp.fit_generator(augment.flow(x,y,batch_size=BATCH),
    epochs=EPOCHS,validation_data=(xtest,ytest))
    model_mlp.save("models/model_mlp_1.h5")
    plotting(history,"models/mlp_graph_1.png")


def main():
    #print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    load_data()


main()
