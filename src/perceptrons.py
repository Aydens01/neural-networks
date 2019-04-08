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
import view as vw
####################################


#############| NOTES |##############
"""
TODO: - clean the program
      - change the activation function
        of the perceptron.
"""
####################################


####################################
############| CLASSES |#############
####################################

class Formal_Neuron():
    """ Formal neuron """
    def __init__(self, size, threshold=0, iterations=200, learning_rate=0.01):
        """ Construct function of the formal neuron

        Parameters :
        ------------
            size {int} : input number \n
            threshold {int} : activation function threshold\n
            classes {int} : sample's division's number\n
            iterations {int} : number of perceptron's training\n
            learning_rate {float} : define the learning rate
        """
        self.threshold = threshold
        self.weights   = np.zeros(size+1)
        self.iterations = iterations
        self.learning_rate = learning_rate
    
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
    
    def train(self, training_data, labels):
        """ Trains the neuron by changing the weights

        Parameters :
        ------------
            training_data {np.array} : the data given to train the 
            neuron\n
            labels {int list} : the expected result for each individual
        Output :
        ------------
            end {bool} : indicates if the neuron has finished the
            training.
        """
        end = False
        n = 0
        while end==False and n<self.iterations:
            end = True
            n += 1
            for (ipt, lbl) in zip(training_data, labels):
                prediction = self.activation(ipt)
                # WARNING: ipt is an np.array
                self.weights[1:] += self.learning_rate*(lbl-prediction)*ipt
                self.weights[0]  += self.learning_rate*(lbl-prediction)
                if prediction != lbl:
                    # print the wrong classified data
                    # print(ipt)
                    end = False
        return(end)

    def __str__(self):
        """ Defines print function for the formal neuron
        """
        threshold = str(np.round(self.weights[0], 3))
        model = "Neuron scheme\nthreshold : "+threshold+"\n"
        for weight in self.weights[1:]:
            model += "-| {} |-\n".format(np.round(weight, 3))
        return(model)

class Perceptron():
    """Perceptron with one layer"""
    def __init__(self, size, classes=2):
        """ Construct function of the perceptron

        Parameters :
        ------------
            size {int} : input number \n
            classes {int} : number of classes
        """
        self.size = size
        self.classes = classes 
        self.network = self.deploy(classes)
    
    def deploy(self, classes):
        """ Initialize the neuronal network

        Parameters :
        ------------
            classes {int} : number of classes
        Output :
        --------
            output {Formal_Neuron list}
        """
        network = []
        i = 0
        while (2**i)<classes:
            i+=1
        for _ in range(i):
            network.append(Formal_Neuron(self.size))
        return(network)

    def activation(self, inputs):
        """ Activation function of the perceptron

        Parameters :
        ------------
            inputs {np.array} : perceptron inputs
        Output :
        --------
            output {list}
        """
        output = []
        for neuron in self.network:
            output.append(neuron.activation(inputs))

        return(output)

    def train(self, training_data, labels):
        """ Trains the perceptron by changing the 
        neuron's weights of the network

        Parameters :
        ------------
            training_data {np.array} : the data given 
            to train the perceptron\n
            labels {int list list} : the expected result
            each individual
        Output :
        --------
            end {bool} : indicates if the perceptron has finished the
            training.
        """
        end = True #FIXME: change init value
        for i in range(len(self.network)):
            lbl = [label[i] for label in labels]
            end *= self.network[i].train(training_data, lbl)
        return(end)

    def __repr__(self):
        """ Defines print function for the perceptron
        """
        model = "------------------\nPERCEPTRON network\n------------------\n"
        for neuron in self.network:
            model += str(neuron)
        return(model)

####################################
############| PROGRAM |#############
####################################

if __name__=="__main__":

    #>>> DEV TESTS <<<#
    p = Perceptron(3, 4)
    print(p)
    
    training_data = [np.array([255, 0, 0]),
                     np.array([0, 255, 0]),
                     np.array([0, 0, 255]),
                     np.array([0, 0, 0])]

    labels = [[0, 0],[0, 1],[1, 0],[1, 1]]

    p.train(training_data, labels)
    print(p)
    print(p.activation(np.array([255, 0, 0])))
    print(p.activation(np.array([0, 255, 0])))
    print(p.activation(np.array([0, 0, 255])))
    print(p.activation(np.array([0, 0, 0])))
    #print(p.activation(np.array([255, 255, 255])))

    print("Approximations")

    print(p.activation(np.array([15, 0, 0])))
    print(p.activation(np.array([0, 200, 0])))
    print(p.activation(np.array([0, 0, 190])))
    print(p.activation(np.array([0, 0, 0])))

