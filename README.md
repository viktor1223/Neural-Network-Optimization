# Description

Traditional neural networks are trained using back propagation. Back propagation computes the gradient the network loss to help automate the learning process of the model. When training a model using back propagation the programmer must explicitly define the networks architecture. This repository explores the effects of automating the creation of a neural network architecture using a genetic algorithm while still training the model using back propagation.

## limitations of project

This repository is currently limited in its ability to create architectures for machine learning. Currently the repository is able to create fully connected neural networks with 1-N hidden layers, 1-MNodes where the number of nodes may differ for each layer. Each layer may use it&#39;s own activation from the following list: sigmoid, tanh, relu, elu. The model may also choose a constant learning rate and dropout rate. The learning rate varies from .1 to 1e-7 by a magnitude of 10 for each step. The dropout rate is within the range of 0 to .7 by steps of 0.1.

## Format of ANN bit string

The genetic algorithm uses something called a chromosome to generate offspring. This chromosome is a binary value representing the models architecture.

User input for the number of hidden layers and number of nodes for each layer will be converted to their binary value. The length of this binary value will then be used to fine the max unsigned binary value that can be represented.

For example if a user enters 5 the binary value is 0101. All zeros within the binary string will be converted to 1 to find the max number represented, in this case 15, or in binary 1111.

Global variable arrays are defined to hold the values of the learning rate (length of 7) , activation function (length of 4), and dropout rate (length of 7). The length of these arrays defines the number of binary characters needed within the models chromosome. Currently 7 and 4 were chosen because the are the max unsigned binary values that can be reparented for their length.

| **Model Parameter** | **Used** | **Number of Hidden Layers** | **Number of Nodes Per Layer** | **Learning Rate** | **Activation Function Per Layer** | **Dropout Rate** |
| --- | --- | --- | --- | --- | --- | --- |
| **Bit size in chromosome** | 1 | MUB(hidden Layer) | MUS(Nodes) | Len(learning rate array) | Len(activation function array) | Len(dropout rate array) |

**Table 1: Chromosome Binary Breakdown**

MUB defined the **M** ax **U** nsigned **B** inary value of the user input of X

Len denotes the length of an array

## Outputs

The repository will output a accuracy per generation graph and a Jason file of the model architecture. These can be found in the &quot;./output/{date}&quot; folder.

## Flow chart


![NNOGA_Flow_Cart](https://github.com/viktor1223/Neural-Network-Optimization/blob/master/Figures/NNOGA_Flow_Chart.jpg?raw=true)

# Usage

The following lines of code in main.py can be changed to increase performance

POPULATION\_SIZE The number of models to train per generation

MUTATION\_CHANCE Chance of randomly tweaking the models chromosome

CROSSOVER\_CHANCE Chance for uniform crossover between two parents to occure

KEEP The percentage of most fit models per generation to keep

maxGen Number of generations to train for

# Future work

Add GUI for

User input on global variables

choose dataset

import custom dataset

Add CNN option for classification and localization

# Project Status

Development has slowed down

# Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change