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

        #print(training_info)

        self.accuracy = score[1]

        return score[1]
