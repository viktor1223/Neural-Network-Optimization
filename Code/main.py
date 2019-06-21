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
learning_rate = [.01, .1, .2, .25]
dropout_rate = [.1, .2, .3, .4, .5]
chanels = [8, 16, 32, 64]
CNN_array = [1, 2, 4, 3] #Kernel, Strides, Pool_Array, Pool_Strides_Array
input_shape = (28,28,1)
val_split = [.1, .2, .3]

outputLayer = None
acc_TOL = 1
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

    """
    Figure out CNN Math to avoid Negative Dimention error and a way to check
    and create a model that will avoid this
    """
    CNN_kernel_array = generate_list_tuple(CNN_numConvLayers, len(CNN_array)) #[2, 4, 6, 10]
    print(CNN_kernel_array)
    i = 0
    list = []
    for i in range(len(CNN_kernel_array)):

        if CNN_kernel_array[i] == (1, 1):
            list.append((CNN_array[0], CNN_array[0]))
        if CNN_kernel_array[i] == (2, 2):
            list.append((CNN_array[1], CNN_array[1]))
        if CNN_kernel_array[i] == (3, 3):
            list.append((CNN_array[2], CNN_array[2]))
        if CNN_kernel_array[i] == (4, 4):
            list.append((CNN_array[3], CNN_array[3]))
    list.sort(reverse=True)
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
    list.sort(reverse=True)
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
    list.sort(reverse=True)
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
    list.sort(reverse=True)
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
    child = networkModle.Model(artificalNN, convolutionNN, xTrain, yTrain,
                            num_classes, MODEL_randNumSeed,
                            MODEL_epochs, batchSize=MODEL_batchSize, optimizer=MODEL_optimizer, cost=MODEL_cost, one_hot=one_hot)
    #Train Model
    print(artificalNN.get_info())
    print(convolutionNN.get_info())

    condition_statment = child.create_architecture()

    if condition_statment == True:
        cost, acc = chil.train(val_split = MODEL_val_split)
    else:
        acc = 0


    print(acc)
    counter = 0



    while acc < acc_TOL:
        counter += 1

        #mutate
        convolutionNN = mutate_CNN(CNN_numConvLayers, CNN_activationFunction_perLayer, CNN_chanels, CNN_kernel_array,
                            CNN_strides_array, CNN_pool_array, CNN_pool_strides_array, CNN_used)
        artificalNN = mutate_ANN(ANN_numHiddenLayers, ANN_numNodes, ANN_learningRate, ANN_activationFunction_perLayer, ANN_dropout_rate, ANN_used)

        child = networkModle.Model(artificalNN, convolutionNN, xTrain, yTrain,
                                num_classes, MODEL_randNumSeed,
                                MODEL_epochs, batchSize=MODEL_batchSize, optimizer=MODEL_optimizer, cost=MODEL_cost, one_hot=one_hot)
        #train model
        condition_statment = child.create_architecture()

        idx = 0
        while (condition_statment == False) and (idx < 1000):
            #mutate CNN parameters
            convolutionNN = mutate_CNN(CNN_numConvLayers, CNN_activationFunction_perLayer, CNN_chanels, CNN_kernel_array,
                                CNN_strides_array, CNN_pool_array, CNN_pool_strides_array, CNN_used)
            #print(convolutionNN.get_info())
            child = networkModle.Model(artificalNN, convolutionNN, xTrain, yTrain,
                                    num_classes, MODEL_randNumSeed,
                                    MODEL_epochs, batchSize=MODEL_batchSize, optimizer=MODEL_optimizer, cost=MODEL_cost, one_hot=one_hot)

            #train model
            condition_statment = child.create_architecture()
            print(condition_statment)
            idx += 1

        cost, acc = child.train(val_split = MODEL_val_split)
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

"""
We're getting a problem where the output is sometimes [6, 3, (1,1)] so we need to
update our problem to define a new list then over write the old list to avoid the (1,1)
We also need to address that either the random number generator or the genetic algorithm
is defining values larger than we want ie we want a range of binary 00 -> 11 (0 -> 3)
"""
def mutate_CNN(CNN_numConvLayers, CNN_activationFunction_perLayer, CNN_chanels, CNN_kernel_array,
                    CNN_strides_array, CNN_pool_array, CNN_pool_strides_array, CNN_used):
    if CNN_used == True:
        CNN_used = int(GA.mutate(bi(1)), 2)
    if CNN_used == False:
        CNN_used = int(GA.mutate(bi(0)), 2)

    if CNN_used == 1:
        CNN_used = True
    if CNN_used == 0:
        CNN_used = False

    CNN_numConvLayers = int(GA.mutate(bi(CNN_numConvLayers)), 2)

    #CNN_activationFunction_perLayer = generate_list(CNN_numConvLayers, len(activation_function)) #make a function to generate list based off how large numConvLayers is
    i = 0
    for i in range(len(range(CNN_numConvLayers))):
        try:
            if CNN_activationFunction_perLayer[i] == activation_function[0]:
                CNN_activationFunction_perLayer[i] = int(GA.mutate(bi(1)),2)
            if CNN_activationFunction_perLayer[i] == activation_function[1]:
                CNN_activationFunction_perLayer[i] = int(GA.mutate(bi(2)),2)
            if CNN_activationFunction_perLayer[i] == activation_function[2]:
                CNN_activationFunction_perLayer[i] = int(GA.mutate(bi(3)),2)
        except:
            CNN_activationFunction_perLayer.append(int(GA.mutate(bi(random.randint(0,3))),2))

    i = 0
    list = []
    for i in range(len(range(CNN_numConvLayers))):
        if CNN_activationFunction_perLayer[i] == 1:
            list.append(activation_function[0])
        if CNN_activationFunction_perLayer[i] == 2:
            list.append(activation_function[1])
        if CNN_activationFunction_perLayer[i] == 3:
            list.append(activation_function[2])
    CNN_activationFunction_perLayer =list

    #make a function to generate list based off how large numConvLayers is
    #CNN_chanels = generate_list(CNN_numConvLayers, len(chanels)) #[16, 32, 64, 128]
    i = 0
    for i in range(len(range(CNN_numConvLayers))):
        try:
            if CNN_chanels[i] == chanels[0]:
                CNN_chanels[i] = int(GA.mutate(bi(1)),2)
            if CNN_chanels[i] == chanels[1]:
                CNN_chanels[i] = int(GA.mutate(bi(2)),2)
            if CNN_chanels[i] == chanels[2]:
                CNN_chanels[i] = int(GA.mutate(bi(3)),2)
            if CNN_chanels[i] == chanels[3]:
                CNN_chanels[i] = int(GA.mutate(bi(4)),2)
        except:
            CNN_chanels.append(int(bin(random.randint(0,3)), 2))

    i = 0
    list = []
    for i in range(len(CNN_chanels)):
        if CNN_chanels[i] == 0:
            list.append(chanels[0])
        if CNN_chanels[i] == 1:
            list.append(chanels[1])
        if CNN_chanels[i] == 2:
            list.append(chanels[2])
        if CNN_chanels[i] == 3:
            list.append(chanels[3])
    CNN_chanels = list

    CNN_kernel_array = mutate_tuple(CNN_kernel_array, CNN_numConvLayers, mutate_max=4)
    CNN_kernel_array.sort(reverse=True)

    CNN_strides_array = mutate_tuple(CNN_strides_array, CNN_numConvLayers, mutate_max=4)
    CNN_strides_array.sort(reverse=True)

    CNN_pool_array = mutate_tuple(CNN_pool_array, CNN_numConvLayers, mutate_max=4)
    CNN_pool_array.sort(reverse=True)

    CNN_pool_strides_array = mutate_tuple(CNN_pool_strides_array, CNN_numConvLayers, mutate_max=4)
    CNN_pool_strides_array.sort(reverse=True)

    convolutionNN = networks.CNN(CNN_numConvLayers,input_shape, CNN_activationFunction_perLayer, CNN_chanels, CNN_kernel_array,
                        CNN_strides_array, CNN_pool_array, CNN_pool_strides_array, used=CNN_used)
    return convolutionNN

def mutate_tuple(input_tuple, iter, mutate_max = 4):
    list_1 = []
    for i in range(iter):
        try:
            tupel = input_tuple[i][0]
            mutated_value = int(GA.mutate(bi(tupel)),2)
            while mutated_value == 0:
                mutated_value = int(GA.mutate(bi(tupel)),2)
            list_1.append((mutated_value, mutated_value))
        except:
            mutated_value = int(GA.mutate(bi(random.randint(1,mutate_max))),2)
            while mutated_value == 0:
                mutated_value = int(GA.mutate(bi(random.randint(1,mutate_max))),2)
            list_1.append((mutated_value, mutated_value))

    print(list_1)

    return list_1

def mutate_ANN(ANN_numHiddenLayers, ANN_numNodes, ANN_learningRate, ANN_activationFunction_perLayer, ANN_dropout_rate, ANN_used, mutate_max=100):
    ANN_used = True #Boolean True or False
    if ANN_used == True:
        CNN_used = int(GA.mutate(bi(1)), 2)
    if CNN_used == False:
        ANN_used = int(GA.mutate(bi(0)), 2)

    if ANN_used == 1:
        ANN_used = True
    if ANN_used == 0:
        ANN_used = False

    ANN_numHiddenLayers = int(GA.mutate(bi(ANN_numHiddenLayers)), 2)

    list = []
    for i in range(ANN_numHiddenLayers):
        try:
            list.append(int(GA.mutate(bi(ANN_numNodes[i])), 2))
        except:
            list.append(int(GA.mutate(bi(random.randint(0,mutate_max))), 2))
    ANN_numNodes = list

    try:
        ANN_numNodes[ANN_numHiddenLayers-1] = outputLayer
    except:
        ANN_numNodes[ANN_numHiddenLayers] = outputLayer
    #these will need to be mapped from bindary to a predefined variables

    idx = None
    mutated_learningRate = None
    for i in range(len(learning_rate)):
        if ANN_learningRate == learning_rate[i]:
            mutated_learningRate = int(GA.mutate(bi(i)), 2)
            idx = mutated_learningRate

    if idx == None:
        idx = int(max_bi(len(learning_rate)), 2)

    for i in range(idx):
        try:
            if mutated_learningRate == i:
                ANN_learningRate = learning_rate[i]
        except:
            if mutated_learningRate == i:
                ANN_learningRate = learning_rate[len(learning_rate)] + random.uniform(0, 0.2)

    #make a functio nto generate a list 1-3 based off how many hidden layers we have
    ANN_activationFunction_perLayer = generate_list(ANN_numHiddenLayers, len(activation_function)) #[ 'sigmoid', 'tanh', 'relu']

    activation_function_list = []
    for i in range(len(ANN_activationFunction_perLayer)):
        for j in range(len(activation_function)):
            if ANN_activationFunction_perLayer[i] == activation_function[j]:
                activation_function_list.append(int(GA.mutate(bi(i)), 2))

    i = 0
    list = []
    for i in range(ANN_numHiddenLayers):
        for j in range(len(activation_function)):
            try:
                if activation_function_list[i] == j:
                    list.append(activation_function[j])
            except:
                idx = random.randint(0,len(activation_function)-1)

                list.append(activation_function[idx])
    ANN_activationFunction_perLayer = list

    for i in range(len(dropout_rate)):
        if dropout_rate[i] == ANN_dropout_rate:
            ANN_dropout_rate = dropout_rate[int(GA.mutate(bi(i)), 2)]

    artificalNN = networks.ANN(ANN_numHiddenLayers, ANN_numNodes, ANN_learningRate, ANN_activationFunction_perLayer, ANN_dropout_rate, ANN_used)

    return artificalNN

def max_bi(binary):
    s = ''
    for i in range(binary):
        s = '1' + s
    return s

main()
exit()
