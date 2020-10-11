import genetic_algorithm as GA
import Utilities as TOOLS
import Networks

import tensorflow as tf
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical

import matplotlib.pyplot as plt 
from datetime import date
from operator import itemgetter
import os

#Global Variables
GENE_SET = '10'             #Binary values for model chomosome Constant DO NOT CHANGE
MAX_NUM_NODES = int(TOOLS.max_bi(len(TOOLS.int_to_bit(50))), 2)                         #Max number of nodes in each layer
MAX_NUM_HIDDEN_LAYERS = int(TOOLS.max_bi(len(TOOLS.int_to_bit(10))), 2)                 #Max number of hidden layers 
TOOLS.global_variables([ 'sigmoid', 'tanh', 'relu', 'elu'],                             #Activation Function 
                        [.1, .01, .001, .0001, .00001, .000001, .0000001, .00000001],   #Learning Rate
                        [0, .1, .2, .3, .4, .5, .6, .7])                                #Dropout Rate
#Change for increased performance 
POPULATION_SIZE = 10        #zeo index
MUTATION_CHANCE = 0.2       #Chance of randomly tweaking the models chromosome
CROSSOVER_CHANCE = 0.45     #Chance for uniform crossover between two parents to occure
KEEP = 0.4                  #Keep the top X percent of the population 
maxGen = 10              #Number of generations


 

ga = GA.Progeny(GENE_SET, MUTATION_CHANCE, CROSSOVER_CHANCE, KEEP)     #Init Genetic Algorithm object 


def get_cifar10():
    """
    Retrieve the CIFAR dataset and process the data.
     return 
     nb_classes         Number of Classes           10
     batch_size         Batch Size                  64
     input_shape        input shape for network     (3072, )
     x_train            Training Data 
     x_test             Test Data 
     y_train            Training Labels
     y_test             Test Labels
    """
    # Set defaults.
    nb_classes = 10
    batch_size = 64
    input_shape = (3072,)

    # Get the data.
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.reshape(50000, 3072)
    x_test = x_test.reshape(10000, 3072)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)

def train(bit_model, input_shape, classes, x_train, x_test, y_train, y_test, batch_size):
    """
    Train each model within the generation 

    return 
    [bit_model, acc, model]
    bit_model       String of binary 1/0 for offspring generation 
    Acc             Accuracy of the model for fitness calculation 
    Model           Keras model for saving the json file 
    """
    optimizer = 'adam'#Can be optimized wi, x_train, x_test, y_train, y_testth GA
    cost = 'mse'
    epochs = 100

    #Find out how to use the learning rate
    num_hidden_layers, num_nodes_per_layer, learning_rate, activation_function_per_layer, dropout_rate, used = TOOLS.ANN_bit_to_model(bit_model, MAX_NUM_HIDDEN_LAYERS, MAX_NUM_NODES, classes)

    network = Networks.Model()
    model = network.create_architecture(num_hidden_layers, num_nodes_per_layer, activation_function_per_layer, dropout_rate, input_shape, optimizer, cost)
    acc = network.train(model, x_train, x_test, y_train, y_test, batch_size, epochs)
    #print(model)
    #model.save_weights("model.h5")
    #print("Saved model to disk")
    #exit()
    return [bit_model, acc, model]


def main():
    """
    main function to control data flow

    savePath        output/(today's data)
    output files 
        model json
        generation vs acc graph 
    """
    today = date.today()
    savePath = "output/"+str(today)+"/"
    if not os.path.exists(savePath):
        os.makedirs(savePath)

    #import/normalize data
    classes, batch_size, input_shape, x_train, x_test, y_train, y_test = get_cifar10()
    #generate bit mondel
    bit_model = TOOLS.ANN_bit(MAX_NUM_HIDDEN_LAYERS, MAX_NUM_NODES)
    #generate population
    population = ga.population(POPULATION_SIZE, len(bit_model))

    graph_list = []
    gen_list = []
    acc, best_acc, counter = 0, 0, 0
    while counter < maxGen:
        #train/calc fitness of population
        print("Counter:\t", counter)
        print("Best Acc:\t", best_acc)
        network_list = []
        for i in range(len(population)):
            network = train(population[i], input_shape, classes, x_train, x_test, y_train, y_test, batch_size)
            network_list.append(network)

            if counter+1 == maxGen:
                #sort by most accurate to least
                graded = sorted(network_list, key=itemgetter(1), reverse=True)

        #evolve
        population, best_acc = ga.Offspring(network_list)
        graph_list.append(best_acc)
        gen_list.append(counter+1)

        counter += 1

    
    # serialize model to JSON
    model_json = graded[0][2].to_json()
    with open(savePath+"model_"+str(today)+".json", "w") as json_file:
        json_file.write(model_json)
    
   #Plot Geneartion vs Accuracy graph 
    plt.plot(gen_list, graph_list)
    plt.xlabel("Generation")
    plt.ylabel("Accuracy")
    plt.title("Generational Accuracy of Model Offspring")
    plt.savefig(savePath+"Model Acc by Generation "+str(today)+".jpg")
    #plt.show()

main()
