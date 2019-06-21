import genetic_algorithm as ga
import model as networkModle
import Networks as networks

import random
from keras.datasets import mnist
from keras.utils import to_categorical

"""
Global variables
"""
activation_function_array = [ 'sigmoid', 'tanh', 'relu', 'elu']
learning_rate_array = [.1, .01, .001, .0001, .00001, .000001, .0000001, .00000001]
dropout_rate_array = [0, .1, .2, .3, .4, .5, .6, .7]
chanels = [8, 16, 32, 64]
CNN_array = [1, 2, 4, 3] #Kernel, Strides, Pool_Array, Pool_Strides_Array
input_shape = (28,28,1)
val_split = [.1, .2, .3]

counter_max = 100
acc_TOL = 0.95
comp_TOL = None
stopping_test = 10
"""
Define Genetic Algorithm Objects
"""
gene_set = '10'
GA = ga.Progeny(gene_set)

def main():

    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    #xTrain = xTrain.reshape(60000,28,28,1)
    #xTest = xTest.reshape(10000,28,28,1)
    yTrain = to_categorical(yTrain)
    yTest = to_categorical(yTest)
    Label_m, Label_n,  = yTrain.shape
    num_classes = Label_n

    #Convert all to binary
    """
    Define Inital Model Parameters

    Define max sizes of ANN_numHiddenLayers and CNN_numConvLayers to define
    length of genes. Use generate_parent to create an inital model for both. A function
    will be needed to parse the genes into usable parameters.

    Need to look out for CNN layer to check math.

    First step get ANN layer working with the model
    """
    num_hidden_layers = 5
    num_nodes=50
    max_num_hidden_layers = int(max_bi(int_to_bit(num_hidden_layers)), 2)
    max_num_nodes = int(max_bi(int_to_bit(num_nodes)), 2)


    ANN_bit_parameters = ANN_bit(max_num_hidden_layers, max_num_nodes)
    print(ANN_bit_parameters)
    ANN_model = ANN_bit_to_model(ANN_bit_parameters, max_num_hidden_layers, max_num_nodes, num_classes)
    #data, labels, classes, randNumSeed, epochs, batchSize, optimizer, cost, ANN, CNN=False, one_hot=False

    MODEL_epochs = 4000
    MODEL_batchSize = 1
    MODEL_randNumSeed = random.randint(1,100)

    #Look into how these change the accuracy
    #they make make it appear more accurate than it is
    #It also might not matter
    MODEL_optimizer = 'adam'
    MODEL_cost = 'mse'
    MODEL_VAL_SPLIT = 0.2

    parent = networkModle.Model( xTrain, yTrain, num_classes, MODEL_randNumSeed, MODEL_epochs, batchSize=MODEL_batchSize, optimizer=MODEL_optimizer, cost=MODEL_cost, ANN=ANN_model)
    print("Create Architecture")
    parent.create_architecture()
    print("Created!")
    cost, acc = parent.train(val_split = MODEL_VAL_SPLIT)
    print("Cost\t\t|\tAccuracy")
    print("{0:0.5f}%\t|\t{1:0.5f}%".format(cost*100, acc*100))
    acc_list = [acc]
    counter_list = [0]
    counter = 1

    while (acc < acc_TOL) or (counter < counter_max):
        print("-----------------------------------------\n\n\n")
        print("Model\t", counter)
        #xTrain = xTrain.reshape(60000,28,28)
        #Mutate
        ANN_bit_parameters = GA.mutate(ANN_bit_parameters)
        print("*****************************************")
        print("ANN BIT PARAMETERS")
        print(ANN_bit_parameters)
        print("*****************************************")
        ANN_model = ANN_bit_to_model(ANN_bit_parameters, max_num_hidden_layers, max_num_nodes, num_classes)
        #Fitness
        child = networkModle.Model( xTrain, yTrain, num_classes, MODEL_randNumSeed, MODEL_epochs, batchSize=MODEL_batchSize, optimizer=MODEL_optimizer, cost=MODEL_cost, ANN=ANN_model)
        child.create_architecture()
        cost, acc = parent.train(val_split = MODEL_VAL_SPLIT)
        print("Cost\t\t|\tAccuracy")
        print("{0:0.5f}%\t|\t{1:0.5f}%".format(cost*100, acc*100))
        acc_list.append(acc)
        counter_list.append(counter)
        #compare acc for most accurate
        counter += 1




def int_to_bit(i):
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

def max_bi(binary):
    s = ''
    for i in range(binary):
        s = '1' + s
    return s

def ANN_bit(num, max_num_nodes):
    """
    Parameters                      number of bits
    used                            1
    num_hidden_layers               num
    num_nodes_per_layer             max_num_nodes  >> List num size
    learning_rate                   len(learning_rate)
    activationFunction_per_layer    len(activation_function)
    dropout_rate                    len(dropout_rate)
    """
    used = '1' #The ANN layer will ALWAYS be used!
    #str(random.randint(0,1))
    #print(used)
    num_hidden_layers = int_to_bit(num)
    #print(num_hidden_layers)
    num_nodes_per_layer = ''
    for i in range(num):
        num_nodes_per_layer += int_to_bit(max_num_nodes)
    #print(len(int_to_bit(max_num_nodes)))
    #print(num_nodes_per_layer)
    learning_rate = int_to_bit(len(learning_rate_array)-1)
    #print(learning_rate)
    activationFunction_per_layer = ''
    for i in range(num):
        activationFunction_per_layer += int_to_bit(len(activation_function_array)-1)
    #print(activationFunction_per_layer)

    dropout_rate = int_to_bit(len(dropout_rate_array)-1)
    #print(dropout_rate)
    #exit()
    parameter = used+num_hidden_layers+num_nodes_per_layer+learning_rate+activationFunction_per_layer+dropout_rate
    #print(parameter)
    return parameter

def ANN_bit_to_model(bit, max_num_hidden_layers, max_num_nodes, output_layer):
    """
    Parameters                      starting Bit point                                                                              Ending Bit Point
    used                            0                                                                                               0
    num_hidden_layers               1                                                                                               max_num_hidden_layers
    num_nodes_per_layer             max_num_hidden_layers + 1                                                                       max_num_nodes * max_num_hidden_layers
    learning_rate                   max_num_nodes * max_num_hidden_layers + 1                                                       len(learning_rate) * max_num_nodes * max_num_hidden_layers
    activationFunction_per_layer    len(learning_rate) * max_num_nodes * max_num_hidden_layers + 1                                  len(activation_function) * len(learning_rate) * max_num_nodes * max_num_hidden_layers
    dropout_rate                    len(activation_function) * len(learning_rate) * max_num_nodes * max_num_hidden_layers + 1       len(dropout_rate) * len(activation_function) * len(learning_rate) * max_num_nodes * max_num_hidden_layers
    """
    used = 1
    #print(used)
    #bits 1 > max_num_nodes bit length
    bit_hidden_layers = ''
    bit_length = len(int_to_bit(max_num_hidden_layers))+1
    i = 1
    while i < bit_length:
        bit_hidden_layers += bit[i]
        i += 1
    num_hidden_layers = int(bit_hidden_layers, 2)
    #print(num_hidden_layers)
    num_nodes_per_layer = []

    i = 0
    for i in range(max_num_hidden_layers):
        j = 1
        bit_num_nodes = ''
        while j < (len(int_to_bit(max_num_nodes))+1):
            bit_num_nodes += bit[bit_length]
            bit_length += 1
            j+=1

        num_nodes_per_layer.append(int(bit_num_nodes, 2))

    #print(num_nodes_per_layer)
    num_nodes_per_layer[len(num_nodes_per_layer)-1] = output_layer

    #bit_length = max_num_hidden_layers * (len(int_to_bit(max_num_nodes))+1)
    prvious_math = bit_length

    #print(len(int_to_bit(len(learning_rate_array)-1)))
    bit_learning_rate = ''
    i = 0
    for i in range(len(int_to_bit(len(learning_rate_array)-1))):
        bit_learning_rate += bit[bit_length]
        bit_length += 1

    learning_rate = int(bit_learning_rate, 2)

    i = 0
    for i in range(len(learning_rate_array)):
        if learning_rate == i:
            learning_rate = learning_rate_array[i]

    #print(learning_rate)
    #bit_length = prvious_math + len(int_to_bit(len(activation_function_array))) + 2
    prvious_math = prvious_math + (len(int_to_bit(len(activation_function_array)))*max_num_hidden_layers)

    i = 0
    activation_function_int = []
    for i in range(max_num_hidden_layers):
        bit_activation_function = ''
        j = 0
        while j < len(int_to_bit(len(activation_function_array)-1)):
            bit_activation_function += bit[bit_length]
            bit_length += 1
            j+=1
            #print(bit_activation_function)
        activation_function_int.append(int(bit_activation_function, 2))

    i = 0
    activation_function_per_layer = []
    for i in range(max_num_hidden_layers):
        j = 0
        for j in range(len(activation_function_array)):
            if activation_function_int[i] == j:
                activation_function_per_layer.append(activation_function_array[j])

    #print(activation_function_per_layer)
    i = 0
    bit_dropout_rate = ''

    #bit_length = prvious_math + len(int_to_bit(len(dropout_rate_array)))
    prvious_math = bit_length

    for i in range(len(int_to_bit(len(dropout_rate_array)))-1):
        bit_dropout_rate += bit[bit_length]
        bit_length += 1


    dropout_rate = int(bit_dropout_rate, 2)


    i = 0
    for i in range(len(dropout_rate_array)):
        if dropout_rate == i:
            dropout_rate = dropout_rate_array[i]


    #print(dropout_rate)

    ANN_model = networks.ANN(num_hidden_layers, num_nodes_per_layer, learning_rate,activation_function_per_layer, dropout_rate, used=used )
    return ANN_model

main()
