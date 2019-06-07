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
            "Number of convolutional Layers:\t\t"+str(self.numConvLayers)+"\n"
            "Chanels:\t\t\t\t"+str(self.chanels)+"\n"+\
            "Kernel Size:\t\t\t\t"+str(self.kernel_array)+"\n"
            "Stride Size:\t\t\t\t"+str(self.strides_array)+"\n"
            "Max Pooling:\t\t\t\t"+str(self.pool_array)+"\n"
            "Max Pooling Strides:\t\t\t"+str(self.pool_strides_array)+"\n"
            "Activation Function:\t\t\t"+str(self.activationFunction)+"\n"
            "-----------------------------------------\n")
        return info
