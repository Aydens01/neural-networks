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
"""
####################################


####################################
############| CLASSES |#############
####################################

class Formal_Neuron():
    """ Formal neuron """
    def __init__(self, size, threshold=0, iterations=50, learning_rate=0.01):
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
        """ Train the neuron by changing the weights

        Parameters :
        ------------
            training_data {np.array} : the data given to train the 
            perceptron\n
            labels {int} : the expected result for each individual
        Output :
        ------------
            end {bool} : indicates if the perceptron has finished the
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

    def __print__(self):
        """ Define print function for the formal neuron
        """
        threshold = str(np.round(self.weights[0], 3))
        model = "Neuron scheme\nthreshold : "+threshold+"\n"
        for weight in self.weights[1:]:
            model += "-| {} |-\n".format(weight)
        print(model)

class Perceptron():
    """Perceptron class"""
    def __init__(self, size, classes=2):
        """
        """
        self.size = size
        self.classes = classes 
        self.network = self.deploy(classes)
    
    def deploy(self, classes):
        """ init the network
        """
        network = []
        i = 0
        while (2**i)<classes:
            i+=1
        for _ in range(i):
            network.append(Formal_Neuron(self.size))
        return(network)
    
    def activation(self, inputs):
        """
        """
        output = []
        for neuron in self.network:
            output.append(neuron.activation(inputs))

        return(output)

    def train(self, training_data, labels):
        """
        """
        end = True
        for i in range(len(self.network)):
            lbl = [label[i] for label in labels]
            end *= self.network[i].train(training_data, lbl)
        return(end)

    def __print__(self):
        print("------------------\nPERCEPTRON network\n------------------")
        for neuron in self.network:
            neuron.__print__()

####################################
############| PROGRAM |#############
####################################

if __name__=="__main__":

    #>>> DEV TESTS <<<#

    p = Perceptron(3, 5)
    p.__print__()

    
    training_data = [np.array([255, 0, 0]),
                     np.array([0, 255, 0]),
                     np.array([0, 0, 255]),
                     np.array([0, 0, 0]),
                     np.array([255, 255, 255])]
    labels = [[0, 0, 0],[0, 0, 1],[0, 1, 0],[0, 1, 1], [1, 0, 0]]

    p.train(training_data, labels)
    p.__print__()
    print(p.activation(np.array([255, 0, 0])))
    print(p.activation(np.array([0, 255, 0])))
    print(p.activation(np.array([0, 0, 255])))
    print(p.activation(np.array([0, 0, 0])))
    print(p.activation(np.array([255, 255, 255])))