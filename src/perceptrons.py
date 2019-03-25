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
      - modify the training function 
      to divide a sample in more
      than two classes
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
    def __init__(self, size, classes=2, iterations=100, learning_rate=0.01):
        """ Construct function of the perceptron (simple)

        Parameters :
        ------------
            size {int} : input number\n
            classes {int} : sample's division's number\n
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
            end {bool} : indicates if the perceptron has finished the
            training.
        """
        end = False
        n = 0
        while end==False and n<self.iterations:
            end = True
            n += 1
            for (ipt, thc) in zip(training_data, theorics):
                prediction = self.activation(ipt)
                self.weights[1:] += self.learning_rate*(thc-prediction)*ipt
                self.weights[0]  += self.learning_rate*(thc-prediction)
                if prediction != thc:
                    # print the wrong classified data
                    # print(ipt)
                    end = False
        return(end)

####################################
############| PROGRAM |#############
####################################

if __name__=="__main__":

    # OR function
    def orFct(value):
        if (value[0]==1 or value[1]==1) :
            return(1)
        else:
            return(0)
    
    # AND function
    def andFct(value):
        if (value[0]==1 and value[1]==1) :
            return(1)
        else:
            return(0)

    training_data1 = vw.DataSet2D(orFct, 0, 2, 10)

    training_data2 = vw.DataSet2D(andFct, 0, 2, 10)

    values = [d.value for d in training_data1.data]
    labels = [d.label for d in training_data1.data]
    
    # Perceptron training
    percep = Perceptron(2)
    print("Is perceptron training finished ?", percep.train(values, labels))
    # Perceptron testing
    testing_data = vw.DataSet2D(percep.activation, 0, 2, 50)

    # Graphic view of the test
    v = vw.View(testing_data.data, "A", "B")
    v.__print__()