#! /usr/bin/env python
#-*- coding: utf-8 -*-

"""
author  : Adrien Lafage\n
date    : 03.19
Formal neuron
=============
Provides
    1. A formal neuron

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
      - make some unit tests
"""
####################################


####################################
############| CLASSES |#############
####################################

class Fneuron():
    "Formal neuron"
    def __init__(self, size, weights=np.array([]), threshold=0):
        """ Construct function of the formal neuron

        Parameters :
        ------------
            size {int} : number of inputs\n
            threshold {int} : activation threshold
        """
        self.threshold = threshold
        self.weights = np.zeros(size+1) if len(weights)!=size+1 else weights
    
    def aggregation(self, inputs):
        """ Aggregation function of the formal neuron

        Parameters :
        ------------
            inputs {np.array} : formal neuron
        Output :
        --------
            output {int}
        """
        return(np.dot(inputs, self.weights[1:])+self.weights[0])
    
    def heaviside(self, inputs):
        """ Heaviside activation function\n
        Linked with biological neuron.
        Parameters :
        ------------
            inputs {np.array} : formal neuron inputs
        Output :
        --------
            output {int}
        """
        return(1.0 if self.aggregation(inputs)>=0 else 0.0)
    
    def sigmoid(self, inputs, a):
        """ Sigmoid activation function\n
        Probabilistic interpretation
        Parameters :
        ------------
            inputs {np.array} : formal neuron inputs
        Output :
        --------
            output {int}
        """
        return(1/(1+np.exp(-a*self.aggregation(inputs))))

    def __repr__(self):
        pass

####################################
############| PROGRAM |#############
####################################

if __name__=="__main__":

    #>>> DEV TESTS <<<#
    neuron = Fneuron(2, np.array([10,1,-1]))
    print(neuron.sigmoid(np.array([50, 60]), 1000))

