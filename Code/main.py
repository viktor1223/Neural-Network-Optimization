import genetic_algorithm as ga
import model as networkModle
import Networks as networks

import random
from keras.datasets import mnist
from keras.utils import to_categorical

"""
Global variables
"""
activation_function = [ 'sigmoid', 'tanh', 'relu']
learning_rate = [0.1, .2, .3, .4, .5]
dropout_rate = [0.1, .2, .3, .4, .5]
chanels = [16, 32, 64, 128]
CNN_array = [2, 4, 8, 16] #Kernel, Strides, Pool_Array, Pool_Strides_Array
input_shape = (28,28,1)
val_split = [0.1, .2, .3]

acc_TOL = -1
comp_TOL = None
stopping_test = 10
"""
Define Genetic Algorithm Objects
"""
gene_set = '10'
GA = ga.Progeny(gene_set)

def main():

    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    xTrain = xTrain.reshape(60000,28,28,1)
    xTest = xTest.reshape(10000,28,28,1)
    yTrain = to_categorical(yTrain)
    yTest = to_categorical(yTest)
    Label_m, Label_n,  = yTrain.shape
    num_classes = Label_n
    #Convert all to binary
    """
    Define Inital Model Parameters

    Define max sizes in bits in zeros
    """
    #ANN
    ANN_used = True #Boolean True or False
    ANN_numHiddenLayers = 20 #
    ANN_numNodes = generate_list(ANN_numHiddenLayers, 100) #make a function to generate list based off how large numHiddenLayers is
    outputLayer = num_classes
    ANN_numNodes[ANN_numHiddenLayers-1] = outputLayer
    #these will need to be mapped from bindary to a predefined variables
    ANN_learningRate = 3 # [0.1, .2, .3, .4, .5]
    #make a functio nto generate a list 1-3 based off how many hidden layers we have
    ANN_activationFunction_perLayer = generate_list(ANN_numHiddenLayers, len(activation_function)) #[ 'sigmoid', 'tanh', 'relu']

    i = 0
    list = []
    for i in range(len(ANN_activationFunction_perLayer)):

        if ANN_activationFunction_perLayer[i] == 1:
            #print(1)
            list.append(activation_function[0])
        if ANN_activationFunction_perLayer[i] == 2:
            #print(2)
            list.append(activation_function[1])
        if ANN_activationFunction_perLayer[i] == 3:
            #print(3)
            list.append(activation_function[2])
    ANN_activationFunction_perLayer = list

    ANN_dropout_rate = 0.1 # [0.1, 0.2, .3, .4, .5]
    artificalNN = networks.ANN(ANN_numHiddenLayers, ANN_numNodes, ANN_learningRate, ANN_activationFunction_perLayer, ANN_dropout_rate, ANN_used)
    #numHiddenLayers, numNodes, learningRate, activationFunction, dropout_rate, used=False

    #CNN
    CNN_used = True
    CNN_numConvLayers = 3
    input_shape = (28,28,1) #const defined by input image
    CNN_activationFunction_perLayer = generate_list(CNN_numConvLayers, len(activation_function)) #make a function to generate list based off how large numConvLayers is

    i = 0
    list = []
    for i in range(len(CNN_activationFunction_perLayer)):
        if CNN_activationFunction_perLayer[i] == 1:
            list.append(activation_function[0])
        if CNN_activationFunction_perLayer[i] == 2:
            list.append(activation_function[1])
        if CNN_activationFunction_perLayer[i] == 3:
            list.append(activation_function[2])
    CNN_activationFunction_perLayer =list

    #make a function to generate list based off how large numConvLayers is
    CNN_chanels = generate_list(CNN_numConvLayers, len(chanels)) #[16, 32, 64, 128]

    i = 0
    list = []
    for i in range(len(CNN_chanels)):
        if CNN_chanels[i] == 1:
            list.append(chanels[0])
        if CNN_chanels[i] == 2:
            list.append(chanels[1])
        if CNN_chanels[i] == 3:
            list.append(chanels[2])
        if CNN_chanels[i] == 4:
            list.append(chanels[3])
    CNN_chanels = list

    CNN_kernel_array = generate_list_tuple(CNN_numConvLayers, len(CNN_array)) #[2, 4, 6, 10]
    print(CNN_kernel_array)
    i = 0
    list = []
    for i in range(len(CNN_kernel_array)):
        print(i)
        if CNN_kernel_array[i] == (1, 1):
            list.append((CNN_array[0], CNN_array[0]))
        if CNN_kernel_array[i] == (2, 2):
            list.append((CNN_array[1], CNN_array[1]))
        if CNN_kernel_array[i] == (3, 3):
            list.append((CNN_array[2], CNN_array[2]))
        if CNN_kernel_array[i] == (4, 4):
            list.append((CNN_array[3], CNN_array[3]))
    CNN_kernel_array = list

    CNN_strides_array = generate_list_tuple(CNN_numConvLayers, len(CNN_array))#tuple

    i = 0
    list = []
    for i in range(len(CNN_strides_array)):
        print(i)
        if CNN_strides_array[i] == (1, 1):
            list.append((CNN_array[0], CNN_array[0]))
        if CNN_strides_array[i] == (2, 2):
            list.append((CNN_array[1], CNN_array[1]))
        if CNN_strides_array[i] == (3, 3):
            list.append((CNN_array[2], CNN_array[2]))
        if CNN_strides_array[i] == (4, 4):
            list.append((CNN_array[3], CNN_array[3]))
    CNN_strides_array = list

    CNN_pool_array = generate_list_tuple(CNN_numConvLayers, len(CNN_array)) #tuple

    i = 0
    list = []
    for i in range(len(CNN_pool_array)):
        print(i)
        if CNN_pool_array[i] == (1, 1):
            list.append((CNN_array[0], CNN_array[0]))
        if CNN_pool_array[i] == (2, 2):
            list.append((CNN_array[1], CNN_array[1]))
        if CNN_pool_array[i] == (3, 3):
            list.append((CNN_array[2], CNN_array[2]))
        if CNN_pool_array[i] == (4, 4):
            list.append((CNN_array[3], CNN_array[3]))
    CNN_pool_array = list

    CNN_pool_strides_array = generate_list_tuple(CNN_numConvLayers, len(CNN_array)) #tuple

    i = 0
    list = []
    for i in range(len(CNN_pool_strides_array)):
        print(i)
        if CNN_pool_strides_array[i] == (1, 1):
            list.append((CNN_array[0], CNN_array[0]))
        if CNN_pool_strides_array[i] == (2, 2):
            list.append((CNN_array[1], CNN_array[1]))
        if CNN_pool_strides_array[i] == (3, 3):
            list.append((CNN_array[2], CNN_array[2]))
        if CNN_pool_strides_array[i] == (4, 4):
            list.append((CNN_array[3], CNN_array[3]))
    CNN_pool_strides_array = list

    convolutionNN = networks.CNN(CNN_numConvLayers,input_shape, CNN_activationFunction_perLayer, CNN_chanels, CNN_kernel_array,
                        CNN_strides_array, CNN_pool_array, CNN_pool_strides_array, used=CNN_used)
    #numConvLayers, input_shape, activationFunction,chanels,
    #                kernel_array, strides_array,pool_array,pool_strides_array, used=False

    #Model

    one_hot = True #constant depends on user dataset
    MODEL_epochs = 4000
    MODEL_batchSize = 64
    MODEL_randNumSeed = random.randint(1,100)

    #Look into how these change the accuracy
    #they make make it appear more accurate than it is
    #It also might not matter
    MODEL_optimizer = 'adam'
    MODEL_cost = 'mse'
    MODEL_val_split = 0.2
    network = networkModle.Model(artificalNN, convolutionNN, xTrain, yTrain,
                            num_classes, MODEL_randNumSeed,
                            MODEL_epochs, batchSize=MODEL_batchSize, optimizer=MODEL_optimizer, cost=MODEL_cost, one_hot=one_hot)
    #Train Model
    print(artificalNN.get_info())
    print(convolutionNN.get_info())

    condition_statment = network.create_architecture()

    if condition_statment == True:
        cost, acc = network.train(val_split = MODEL_val_split)
    else:
        acc = 0

    print(acc)
    counter = 0



    while acc < acc_TOL:
        counter += 1

        #mutate

        #train model
        condition_statment = network.create_architecture()

        if condition_statment == True:
            cost, acc = network.train(val_split = MODEL_val_split)
        else:
            break

        #compare for most accurate overall

        #Prevent infinite loop if desired acc can not be reached
        if counter >= stopping_test:
            print("The accuracy desired for this model can not be reached in %d evolutions", counter)
            #save most accurate model
            break



    #save model


def bi(i):
    if i == 0:
        return "0"
    s = ''
    while i:
        if i & 1 == 1:
            s = "1" + s
        else:
            s = "0" + s
        i //= 2
    return s

def generate_list(index, max_num):
    list = []
    for i in range(index):
        list.append(random.randint(1,max_num))

    return list

def generate_list_tuple(index, max_num):
    list = []
    for i in range(index):
        num = random.randint(1,max_num)
        list.append((num,num))
    return list



main()
