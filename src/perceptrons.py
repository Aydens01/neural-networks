#! /usr/bin/env python
#-*- coding: utf-8 -*-

"""
author  : Adrien Lafage\n
date    : 03.19
Perceptrons
=============
Provides
    1. A formal neuron
    2. A perceptron with the Franck Rosenblatt rule

Documentation
-------------
...
"""

############| IMPORTS |#############
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
####################################


#############| NOTES |##############
"""
TODO: clean the program
"""
####################################


####################################
############| CLASSES |#############
####################################

class Formal_Neuron():
    """ Formal neuron """
    def __init__(self, size, threshold=0):
        """ Construct function of the formal neuron

        Parameters :
        ------------
            size {int} : input number \n
            threshold {int} : activation function threshold
        """
        self.threshold = threshold
        self.weights   = np.zeros(size+1)
    
    def aggregation(self, inputs):
        """ Aggregation function of the formal neuron

        Parameters :
        ------------
            inputs {np.array} : formal neuron inputs
        Output :
        --------
            output {int}
        """
        output = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return(output)

    def activation(self, inputs):
        """ Activation function of the formal neuron

        Parameters :
        ------------
            inputs {np.array} : formal neuron inputs 
        Output :
        --------
            output {int}
        """
        value = self.aggregation(inputs)

        if value > self.threshold:
            output = 1
        else :
            output = 0
        
        return(output)

    def __print__(self):
        """ Define print function for the formal neuron
        """
        model = "Neuron scheme\n"
        for weight in self.weights:
            model += "-| {} |-\n".format(weight)
        print(model)

class Perceptron(Formal_Neuron):
    """ Perceptron with the Frank Rosenblatt learning rule"""
    def __init__(self, size, iterations=100, learning_rate=0.01):
        """ Construct function of the perceptron (simple)

        Parameters :
        ------------
            size {int} : input number\n
            iterations {int} : number of perceptron's training\n
            learning_rate {float} : define the learning rate
        """
        Formal_Neuron.__init__(self, size)
        self.iterations = iterations
        self.learning_rate = learning_rate
    
    def train(self, training_data, theorics):
        """ Train the model by changing the weights

        Parameters :
        ------------
            training_data {np.array} : the data given to train the 
            perceptron\n
            theorics {int} : the expected result for each individual
        Output :
        ------------
            end {bool} : indicate if the perceptron has finished to 
            train
        """
        for _ in range(self.iterations):
            for (ipt, thc) in zip(training_data, theorics):
                prediction = self.activation(ipt)
                self.weights[1:] += self.learning_rate*(thc-prediction)*ipt
                self.weights[0]  += self.learning_rate*(thc-prediction)
        
        end = True
        for (ipt, thc) in zip(training_data, theorics):
            if self.activation(ipt) != thc:
                # print the wrong classified data
                # print(ipt)
                end = False
        return(end)

####################################
############| PROGRAM |#############
####################################

if __name__=="__main__":
    # functions
    def color(value):
        if value == 1:
            return("r")
        else:
            return("b")
    # training set initialization
    training_data = np.random.randint(0, 50, size=(500, 2))
    # testing set initialization
    testing_data  = np.random.randint(0, 50, size=(5000, 2))

    # two class definition
    colors_train = []
    theorics = []
    for point in training_data:
        if point[1] >= 2*point[0]+5:
            colors_train.append("r")
            theorics.append(1)
        else:
            colors_train.append("b")
            theorics.append(0)
    
    # init perceptron
    percep = Perceptron(2)
    print("Is perceptron training finished ?", percep.train(training_data, theorics))

    colors_test = []
    for point in testing_data:
        colors_test.append(color(percep.activation(point)))

    # print the training data set
    plt.scatter([point[0] for point in testing_data ], [point[1] for point in testing_data], s=1, c=colors_test)
    plt.show()

    # print the testing data set
    plt.scatter([point[0] for point in training_data ], [point[1] for point in training_data], s=3, c=colors_train)
    plt.show()

    