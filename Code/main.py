import genetic_algorithm as GA
import Utilities as TOOLS
import Networks

from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical

import datetime
import os


file_name = "LOG.csv"

#Global Variables
POPULATION_SIZE = 20 #zeo index

GENE_SET = '10'
MUTATION_CHANCE = 0.6
CROSSOVER_CHANCE = 0.45
KEEP = 0.4
ACC_TOL = 0.90
MAX_NUM_NODES = int(TOOLS.max_bi(len(TOOLS.int_to_bit(100))), 2)
MAX_NUM_HIDDEN_LAYERS = int(TOOLS.max_bi(len(TOOLS.int_to_bit(20))), 2)
TOOLS.global_variables([ 'sigmoid', 'tanh', 'relu', 'elu'],
                        [.1, .01, .001, .0001, .00001, .000001, .0000001, .00000001],
                        [0, .1, .2, .3, .4, .5, .6, .7])

ga = GA.Progeny(GENE_SET, MUTATION_CHANCE, CROSSOVER_CHANCE, KEEP)


def get_cifar10():
    """Retrieve the CIFAR dataset and process the data."""
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
    optimizer = 'adam'#Can be optimized wi, x_train, x_test, y_train, y_testth GA
    cost = 'mse'
    epochs = 100

    #Find out how to use the learning rate
    num_hidden_layers, num_nodes_per_layer, learning_rate, activation_function_per_layer, dropout_rate, used = TOOLS.ANN_bit_to_model(bit_model, MAX_NUM_HIDDEN_LAYERS, MAX_NUM_NODES, classes)

    network = Networks.Model()
    model = network.create_architecture(num_hidden_layers, num_nodes_per_layer, activation_function_per_layer, dropout_rate, input_shape, optimizer, cost)
    acc = network.train(model, x_train, x_test, y_train, y_test, batch_size, epochs)
    return [bit_model, acc]


def main():

    #import/normalize data
    classes, batch_size, input_shape, x_train, x_test, y_train, y_test = get_cifar10()
    #generate bit mondel
    bit_model = TOOLS.ANN_bit(MAX_NUM_HIDDEN_LAYERS, MAX_NUM_NODES)
    #generate population
    population = ga.population(POPULATION_SIZE, len(bit_model))

    #print(population)

    graph_list = []
    acc, best_acc, counter = 0, 0, 0
    while acc < ACC_TOL:

        if counter == 1000:
            break
        #end if

        #train/calc fitness of population
        print("Counter:\t", counter)
        print("Best Acc:\t", best_acc*100, "%")
        network_list = []
        for i in range(len(population)):
            network = train(population[i], input_shape, classes, x_train, x_test, y_train, y_test, batch_size)
            network_list.append(network)

        #evolve
        population, acc = ga.Offspring(network_list)

        if acc > best_acc:
            best_acc = acc
            graph_list.append([best_acc, counter])
            TOOLS.csv_log(file_name, counter, best_acc)
        counter += 1

    #print/log most fit model
    #graph

main()
