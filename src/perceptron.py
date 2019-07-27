#! /usr/bin/env python
#-*- coding: utf-8 -*-

"""
author  : Adrien Lafage\n
date    : 04.19
Perceptron
=============
Provides
    - A perceptron

Documentation
-------------
You can find it here <https://github.com/Aydens01/neural-networks/blob/dev/doc/perceptron.md>
"""

############| IMPORTS |#############
import os
import sys
import numpy as np
####################################


#############| NOTES |##############
"""
todo: __repr__ function
"""
####################################


####################################
############| CLASSES |#############
####################################

class Perceptron():
    "Perceptron"
    def __init__(self, size_in, size_out, weights=np.array([])):
        """ Construct function of the perceptron

        Parameters :
        ------------
            size {int} : number of inputs\n
            size_out {int} :number of outputs\n
            weights {np.matrix} : array of the formal neuron weights
        """
        self.size_in = size_in
        self.size_out = size_out
        self.weights = np.zeros((size_out, size_in+1)) if weights.shape != (size_out, size_in+1) else weights
        
    
    def aggregation(self, inputs):
        """ Aggregation function of the perceptron

        Parameters :
        ------------
            inputs {np.array} : perceptron inputs
        Output :
        --------
            outputs {np.array}
        """
        inputs = np.insert(inputs, 0, 1)
        return(np.dot(self.weights, inputs))

    def heaviside(self, inputs):
        """ Heaviside activation function
        """
        outputs = ''
        agg = self.aggregation(inputs)
        for i in range(self.size_out):
            outputs += '1' if agg[i]>0 else '0'
        # return(outputs)
        # tmp 
        return(int(str(outputs), 2))

    def softmax(self, inputs):
        """ Soft-max activation function
        Probabilistic interpretation for multi-class
        classification
        Parameters :
        ------------
            inputs {np.array} : perceptron inputs
        Output :
        --------
            outputs {np.array}
        """
        outputs = np.zeros(self.size_out)
        agg = self.aggregation(inputs)
        for i in range(self.size_out):
            outputs[i] = np.exp(agg[i])/sum(np.exp(agg))
        return(outputs)
    
    def prediction(self, inputs):
        """ Returns the index of the higher probability
        Parameters :
        ------------
            inputs {np.array} : perceptron inputs
        Output :
        --------
            index {int}
        """
        outputs = self.softmax(inputs)
        #//print(outputs)
        index = np.argmax(outputs)
        
        return(index)

    def __repr__(self):
        """ Informations about the perceptron
        Output :
        --------
            status {str}
        """
        status = "_______________________\n"
        status += "      Perceptron      \n\n"
        #TODO:
        status += "_______________________\n"
        return(status)

####################################
############| PROGRAM |#############
####################################

if __name__=="__main__":

    #>>> DEV TESTS <<<#
    pass
    