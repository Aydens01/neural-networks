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
import fneuron as fn
####################################


#############| NOTES |##############
"""
TODO: __repr__ function
      checking of init parameters
"""
####################################


####################################
############| CLASSES |#############
####################################

class Perceptron():
    "Perceptron"
    def __init__(self, size_in, size_out, fneurons):
        """ Construct function of the perceptron

        Parameters :
        ------------
            size {int} : number of inputs\n
            size_out {int} :number of outputs\n
            fneuron {Fneuron list} : list of formal neuron
        """
        self.size_in = size_in
        self.size_out = size_out
        self.fneurons = fneurons
        
    '''
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
    '''

    def compute(self, inputs, decision_fct=fn.Fneuron.sigmoid):
        """ Computing function of the perceptron
        using formal neurons
        Parameters :
        ------------
            inputs {np.array} : perceptron inputs
        Output :
        --------
            outputs {np.array}
        """
        outputs = np.array([])
        for neuron in self.fneurons:
            outputs = np.append(outputs, decision_fct(neuron, inputs))

        return(outputs)


    def heaviside(self, inputs):
        """ Heaviside activation function
        Parameters : 
        ------------
            inputs {np.array} : perceptron inputs
        Output : 
            output {np.array}
        """
        outputs = np.array([])
        for neuron in self.fneurons:
            outputs = np.append(outputs, neuron.heaviside(inputs))

        return(outputs)
    
    def sigmoid(self, inputs):
        pass

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
        for i in range(len(self.fneurons)):
            outputs[i] = np.exp(self.fneurons[i].aggregation(inputs))
        outputs = outputs/sum(outputs)
        '''
        agg = self.aggregation(inputs)
        for i in range(self.size_out):
            outputs[i] = np.exp(agg[i])/sum(np.exp(agg))
        '''
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
        # print(outputs)
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
        #TODO: display the perceptron's weigths
        status += "_______________________\n"
        return(status)

####################################
############| PROGRAM |#############
####################################

if __name__=="__main__":

    #>>> DEV TESTS <<<#
    p = Perceptron(2, 1, [fn.Fneuron(2, np.array([10, 1, -1]))])
    print(p.heaviside(np.array([0, 100])))
    