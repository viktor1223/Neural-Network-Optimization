<<<<<<< HEAD:Code/V1/Networks.py
"""
Author: Viktor Ciroski
"""

"""
Look into generalizing this to work for CNN not just ANN
    ie find a way to call maxpooling and convolutional layers
        it could be as easy and defining a number of convolutional layers and the convolution sizes
Look into creating an model out of inheaitence! this will help keep our parameters small in the model class
"""
class ANN:

    def __init__(self, numHiddenLayers, numNodes, learningRate, activationFunction, dropout_rate, used=False):

        """
        Things GA can optimize
        """
        self.used = used
        self.numHiddenLayers = numHiddenLayers
        self.numNodes = numNodes
        self.learningRate = learningRate
        self.activationFunction = activationFunction
        self.dropout_rate = dropout_rate


    def get_info(self):

        if self.used == False:
            info = "\nArtificial Neural Network Attributes\n-----------------------------------------\n"+\
                    "Set used=True""\n"+\
                    "-----------------------------------------\n"


        else:
            info = ("\nArtificial Neural Network Attributes\n-----------------------------------------\n"+\
                    "Number Hidden Layers:\t"+str(self.numHiddenLayers)+"\n"+\
                    "Number of Nodes:\t"+str(self.numNodes)+"\n"+\
                    "Learning Rate:\t\t"+str(self.learningRate)+"\n"+\
                    "Activation Function:\t"+str(self.activationFunction)+"\n"+\
                    "Dropout Rate:\t\t"+str(self.dropout_rate)+\
                    "\n-----------------------------------------")
        return info

class CNN:
    def __init__(self, numConvLayers, input_shape, activationFunction,chanels,
                    kernel_array, strides_array,pool_array,pool_strides_array, used=False):
        self.used = used
        self.numConvLayers = numConvLayers
        self.input_shape = input_shape
        self.activationFunction = activationFunction
        self.chanels = chanels
        self.kernel_array = kernel_array
        self.strides_array = strides_array
        self.pool_array = pool_array
        self.pool_strides_array = pool_strides_array

    def get_info(self):
        if self.used == False:
            info = ("\nConvolutional Neural Network Attributes\n-----------------------------------------\n"+\
                    "set used=False""\n"
                    "-----------------------------------------\n")

        else:
            info = ("\nConvolutional Neural Network Attributes\n-----------------------------------------\n"+\
            "Input Shape:\t\t\t"+str(self.input_shape)+"\n"
            "Number of convolutional Layers:\t\t"+str(self.numConvLayers)+"\n"
            "Chanels:\t\t\t\t"+str(self.chanels)+"\n"+\
            "Kernel Size:\t\t\t\t"+str(self.kernel_array)+"\n"
            "Stride Size:\t\t\t\t"+str(self.strides_array)+"\n"
            "Max Pooling:\t\t\t\t"+str(self.pool_array)+"\n"
            "Max Pooling Strides:\t\t\t"+str(self.pool_strides_array)+"\n"
            "Activation Function:\t\t\t"+str(self.activationFunction)+"\n"
            "-----------------------------------------\n")
        return info
=======
"""
Author: Viktor Ciroski
"""
import keras
from keras import models, layers
from keras.layers import Dense, Conv2D, Flatten, InputLayer, MaxPooling2D, Dropout
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
import keras.backend as k_backend

import time

class Model():

    def __init__(self):
        self.accuracy = None

    def create_architecture(self, num_hidden_layers, num_nodes_per_layer, activation_function_per_layer, dropout_rate, input_shape, optimizer, cost):
        #self.log_architecture()



        weight = []
        activation = []
        biases = []
        hiddenLayer = []
        idx = 0



        k_backend.clear_session()
        model = models.Sequential()



        idx = 0
        print("Creating Dense Layers")
        #log.write("Creating Dense Layers\n")
        for idx in range(num_hidden_layers):
            layerOutputSize = num_nodes_per_layer[idx]

            if idx == 0:
                model.add(layers.Dense(layerOutputSize, activation=activation_function_per_layer[idx], input_shape=input_shape))
            else:
                model.add(layers.Dense(layerOutputSize, activation=activation_function_per_layer[idx]))
            model.add(Dropout(dropout_rate))


        model.compile(optimizer=optimizer,
              loss=cost,
              metrics=['accuracy'])
        print("Neural Network Model is created!")
        #log.write("Neural Network Model is created!\n")
        return model

    def train(self, model, x_train, x_test, y_train, y_test, batch_size, epochs, val_split=0.2):

        #To view tensorboard use
        # tensorboard --logdir=/full_path_to_your_logs --host=127.0.0.1
        #tensorboard = TensorBoard(log_dir=folder_name+'graph', histogram_freq=1,
        #                          write_graph=True, write_images=False)
        #tensorboard.set_model(self.model)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

        callbacks = [es]

        #print("Training Model")
        #log.write("Training Model\n")
        start = time.time()
        start_hours, rem = divmod(start, 3600)
        start_minutes, start_seconds = divmod(rem, 60)
        #print("Data shape")
        #print(self.data.shape)


        history = model.fit(x_train,
                            y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_split = val_split,
                            callbacks=callbacks,
                            verbose=0)

        end = time.time()
        #print("Training Completed")
        #log.write("Training Completed\n")
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        training_time = "\n\n\n\n\n"+\
                        "-----------------------------------------\n"+\
                        "Total Training Time:\t\t{:0>2}:{:0>2}:{:05.2f}\n".format(int(hours),int(minutes),seconds)+\
                        "-----------------------------------------"+\
                        "\n\n\n\n\n"
        #print(training_time)



        model.summary()

        score = model.evaluate(x_test, y_test) #[test_loss, test_acc]


        loss = history.history['loss']
        accuracy = history.history['acc']
        val_loss = history.history['val_loss']
        val_acc = history.history['val_acc']

        training_info = training_time+\
                        "Training Loss:\t\t"+str(score[0]*100)+"%\n"+\
                        "Training Accuracy:\t"+str(score[1]*100)+"%\n\n"+\
                        "Validation Loss:\t"+str(val_loss[len(val_loss)-1]*100)+"%\n"+\
                        "Validation Accuracy:\t"+str(val_acc[len(val_acc)-1]*100)+"%\n"

        print(training_info)

        self.accuracy = score[1]

        return score[1]
>>>>>>> master:Code/Networks.py
