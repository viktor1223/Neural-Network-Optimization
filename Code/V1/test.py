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


"""
Define Genetic Algorithm Objects
"""
gene_set = '10'
GA = ga.Progeny(gene_set)

def main():
    counter_max = 1000
    acc_TOL = 0.95
    comp_TOL = None

    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    xTrain = xTrain.reshape(60000, 784)
    xTest = xTest.reshape(10000, 784)
    xTrain = xTrain.astype('float32')
    xTest = xTest.astype('float32')
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


    max_num_hidden_layers = int(max_bi(len(int_to_bit(num_hidden_layers))), 2)
    max_num_nodes = int(max_bi(len(int_to_bit(num_nodes))), 2)


    print("TEST")
    ANN_bit_parameters = ANN_bit(max_num_hidden_layers, max_num_nodes)
    print(len(ANN_bit_parameters))
    #Randomize inital ANN parameters
    ANN_bit_parameters = GA.generate_parent(len(ANN_bit_parameters))
    #print(ANN_bit_parameters)
    #ANN_model = ANN_bit_to_model(ANN_bit_parameters, max_num_hidden_layers, max_num_nodes, num_classes)

    #data, labels, classes, randNumSeed, epochs, batchSize, optimizer, cost, ANN, CNN=False, one_hot=False

    MODEL_epochs = 4000
    MODEL_batchSize = 1
    MODEL_randNumSeed = random.randint(1,100)

    #Look into how these change the accuracy
    #they make make it appear more accurate than it is
    #It also might not matter
    MODEL_optimizer = 'adam'#Can be optimized with GA
    MODEL_cost = 'mse'  #Still need to learn if this would change how we calc the acc
    MODEL_VAL_SPLIT = 0.2
    """
    parent = networkModle.Model( xTrain, yTrain, num_classes, MODEL_randNumSeed, MODEL_epochs, batchSize=MODEL_batchSize, optimizer=MODEL_optimizer, cost=MODEL_cost, ANN=ANN_model)
    print("Create Architecture")
    parent.create_architecture()
    print("Created!")

    cost, acc = parent.train(val_split = MODEL_VAL_SPLIT)
    print("Cost\t\t|\tAccuracy")
    print("{0:0.5f}%\t|\t{1:0.5f}%".format(cost*100, acc*100))
    parent.save_model()
    """
    acc = 0
    most_accurate = acc

    acc_array = []
    acc_list = []
    counter_list = [0]
    counter = 0


    log, file_name = networkModle.get_log_file()


    """
    New function trying to figuer out what above should be in it too
    """

    while ((acc < acc_TOL) and (counter < counter_max)) or ((acc < acc_TOL) or (counter < counter_max)):
        #networkModle.open_log_file(file_name)

        print("-----------------------------------------")
        print("Most Accurate Model:\t"+str(most_accurate*100)+"%")
        print("-----------------------------------------\n\n\n")
        print("Model\t", counter)
        #xTrain = xTrain.reshape(60000,28,28)
        #Mutate
        #add functionality for cross over
        ANN_parameters = GA.mutate(ANN_bit_parameters)
        print("*****************************************")
        print("ANN BIT PARAMETERS")
        print(ANN_parameters)
        print("*****************************************")
        ANN_model = ANN_bit_to_model(ANN_parameters, max_num_hidden_layers, max_num_nodes, num_classes)
        #Fitness
        parent = networkModle.Model( xTrain, yTrain, num_classes, MODEL_randNumSeed, MODEL_epochs, batchSize=MODEL_batchSize, optimizer=MODEL_optimizer, cost=MODEL_cost,input_shape=(784,), ANN=ANN_model)
        print("Create Architecture")
        #parent.create_architecture()
        print("Created!")

        cost, acc = parent.train(val_split = MODEL_VAL_SPLIT)
        print("Cost\t\t|\tAccuracy")
        print("{0:0.5f}%\t|\t{1:0.5f}%".format(cost*100, acc*100))
        acc_list.append(acc)
        counter_list.append(counter)


        #compare acc for most accurate
        if acc >= most_accurate:
            most_accurate = acc #hold acc value
            best_parent = parent      #holds model
            ANN_bit_parameters = ANN_parameters #holds ANN bit parameters
            best_parent.log_architecture()
            #best_parent.log_training_info()
            acc_array.append(most_accurate)



        counter += 1
        #networkModle.close_log_file()




"""
Utils file
"""

main()
