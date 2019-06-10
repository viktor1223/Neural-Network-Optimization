
#Classes Imports

import visualize as visualize

#Model Imports
import tensorflow as tf
import keras
from keras import models, layers
from keras.layers import Dense, Conv2D, Flatten, InputLayer, MaxPooling2D, Dropout
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard

import numpy as np
import random

#Visualization Imports
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#Documentation Imports
import os
import logging
import time
from datetime import date
import calendar


#Global Variables
log_time = time.time()
log_date = str(date.today())

start_hours, rem = divmod(log_time, 3600)
start_minutes, start_seconds = divmod(rem, 60)

print("{:0>2}_{:0>2}".format(int(start_hours),int(start_minutes)))
folder_name = log_date+"_"+str("{:0>2}_{:0>2}_{:05.2f}".format(int(start_hours),int(start_minutes),start_seconds))+"/"
file_name = log_date+"_"+"{:0>2}_{:0>2}".format(int(start_hours),int(start_minutes))

os.mkdir(folder_name)

log = open(folder_name+file_name+".txt", 'w+')

log.write("Start:\n"+folder_name+"\n\n-----------------------------------------\n\n")

class Model():

    def __init__(self,ANN, CNN, data, labels, classes, randNumSeed,
                 epochs, batchSize, optimizer, cost, one_hot=False):

        self.ANN = ANN
        self.CNN = CNN

        self.model = None
        self.data = data
        self.labels = labels
        self.classes = classes
        self.inputNumNodes = None

        self.x = None
        self.y = None
        self.outputLayer = None

        self.one_hot = one_hot
        self.epochs = epochs
        self.batchSize = batchSize
        self.randNumSeed = randNumSeed
        self.rng = np.random.RandomState(self.randNumSeed)
        self.optimizer = optimizer
        self.cost = cost





    def get_info(self):
        info = "\nModel Attributes\n-----------------------------------------\n"+\
                "Artificial Neural Network:\t"+str(self.ANN.used)+"\n"+\
                "Convolution Neural Network:\t"+str(self.CNN.used)+"\n"+\
                "Input Node Number:\t\t"+str(self.inputNumNodes)+"\n"+\
                "Epoches:\t\t\t"+str(self.epochs)+"\n"+\
                "Batch Size:\t\t\t"+str(self.batchSize)+"\n"+\
                "Random Number Seed:\t\t"+str(self.randNumSeed)+"\n"+\
                "Optimizer:\t\t\t"+str(self.optimizer)+"\n"+\
                "Cost:\t\t\t\t"+str(self.cost)+"\n"+\
                "\n-----------------------------------------"
        return info

    def log_architecture(self):
        log.write(self.ANN.get_info()+"\n")
        log.write(self.CNN.get_info()+"\n")
        log.write(self.get_info()+"\n")

    def create_architecture(self):
        self.log_architecture()

        self.x = tf.placeholder(shape=[None, self.inputNumNodes], dtype=tf.float32)
        self.y = tf.placeholder(shape=[None, self.classes],  dtype=tf.float32)
        weight = []
        activation = []
        biases = []
        hiddenLayer = []
        idx = 0

        [M, n] = self.labels.shape

        preLayer = self.x

        self.model = models.Sequential()

        try:
            if self.CNN.used == True:
                firstPass = True
                idx = 0
                for idx in range(self.CNN.numConvLayers):
                    if firstPass == True:
                        print("Creating convolutional Layers")
                        log.write("Creating Convolutional Layers\n")
                        self.model.add(Conv2D(self.CNN.chanels[idx],
                            kernel_size=self.CNN.kernel_array[idx],
                            strides=self.CNN.strides_array[idx],
                            activation=self.CNN.activationFunction[idx],
                            input_shape=self.CNN.input_shape))

                        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
                        firstPass = False
                    else:
                        self.model.add(Conv2D(self.CNN.chanels[idx],
                            kernel_size=self.CNN.kernel_array[idx],
                            strides=self.CNN.strides_array[idx],
                            activation=self.CNN.activationFunction[idx]))
        except:
            log.write("\n\nConvolutional Layer is not possible\n\n")
            return False

        self.model.add(Flatten())

        idx = 0
        print("Creating Dense Layers")
        log.write("Creating Dense Layers\n")
        for idx in range(self.ANN.numHiddenLayers):
            print(idx)
            layerOutputSize = self.ANN.numNodes[idx]

            self.model.add(layers.Dense(layerOutputSize, activation=self.ANN.activationFunction[idx]))
            self.model.add(Dropout(self.ANN.dropout_rate))
            """
                Add Modularity to individualize activation function for each layer
                    and/or each node
            """


        self.model.compile(optimizer=self.optimizer,
              loss=self.cost,#The cost function might be casuing the error !!!!
              metrics=['accuracy'])
        print("Neural Network Model is created!")
        log.write("Neural Network Model is created!\n")
        return None



    def train(self, val_split = None):
        #To view tensorboard use
        # tensorboard --logdir=/full_path_to_your_logs --host=127.0.0.1
        tensorboard = TensorBoard(log_dir=folder_name+'graph', histogram_freq=1,
                                  write_graph=True, write_images=False)
        tensorboard.set_model(self.model)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

        callbacks = [tensorboard, es]

        print("Training Model")
        log.write("Training Model\n")
        start = time.time()
        start_hours, rem = divmod(start, 3600)
        start_minutes, start_seconds = divmod(rem, 60)

        history = self.model.fit(self.data, self.labels,
                    batch_size=self.batchSize,
                    epochs=self.epochs,
                    validation_split = val_split,
                    callbacks=callbacks,
                    verbose=0)

        end = time.time()
        print("Training Completed")
        log.write("Training Completed\n")
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Total Training Time:\t\t{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
        log.write("\n\n\n\n\n")
        log.write("-----------------------------------------\n")
        log.write("Total Training Time:\t\t{:0>2}:{:0>2}:{:05.2f}\n".format(int(hours),int(minutes),seconds))
        log.write("-----------------------------------------")
        log.write("\n\n\n\n\n")

        self.model.summary()

        score = self.model.evaluate(self.data, self.labels) #[test_loss, test_acc]

        self.model.save(folder_name+"{:0>2}_{:0>2}_{:05.2f}".format(int(start_hours),int(start_minutes),start_seconds)+".h5")#Format to normal time!
        loss = history.history['loss']
        accuracy = history.history['acc']
        val_loss = history.history['val_loss']
        val_acc = history.history['val_acc']

        log.write("Training Loss:\t\t"+str(score[0])+"\n")
        log.write("Training Accuracy:\t"+str(score[1])+"\n\n")
        log.write("Validation Loss:\t"+str(val_loss[len(val_loss)-1])+"\n")
        log.write("Validation Accuracy:\t"+str(val_acc[len(val_acc)-1])+"\n")

        return score

    def test(self, xTest, yTest, batch_size):
        score, acc = self.model.evaluate(xTest, yTest,
                            batch_size=batch_size)
        print('Test score:', score)
        print('Test accuracy:', acc)

        log.write("Test Score:\t\t"+str(score)+"\n")
        log.write("Test Accuracy:\t\t"+str(acc))

        return None
