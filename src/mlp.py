#! /usr/bin/env python
#-*- coding: utf-8 -*-

"""
author  : Adrien Lafage\n
date    : 04.19
Multi-layer perceptron
=============
Provides
    - A multi-layer perceptron

Documentation
-------------
You can find it here <https://github.com/Aydens01/neural-networks/blob/dev/doc/mlp.md>
"""

############| IMPORTS |#############
import os
import sys
import numpy as np
import fneuron as fn
####################################


#############| NOTES |##############
"""
todo: __repr__ function
"""
####################################


####################################
############| CLASSES |#############
####################################
class Mlp():
    "Multi-layer perceptron"
    def __init__(self, size_in, size_out, layers_sizes, layers):
        self.size_in = size_in
        self.size_out = size_out
        self.layers_sizes = layers_sizes
        self.layers = layers
    
    ##################################
    # Some decision functions provides 
    # with the Mlp object
    ##################################
    def aggregation(self, inputs):
        # set up
        output_network = [inputs]
        tmp = inputs

        for layer in self.layers :
            output_layer = np.array([])
            for neuron in layer :
                output_layer = np.append(output_layer, neuron.sigmoid(tmp))
            tmp = output_layer
            output_network.append(output_layer)
        return(output_network)


####################################
############| PROGRAM |#############
####################################

if __name__=="__main__":

    #>>> DEV TESTS <<<#
    # test = Mlp(2, 1, [3], [np.array([[-10, 1, 0],[45, -1, -1],[-10, 0, 1]])])
    w1 = np.array([-1, 2, 2]) # OR logic
    w2 = np.array([3, -2, -2]) # NAND logic
    w3 = np.array([-3, 2, 2]) # AND logic
    mlp = Mlp(2, 1, [2, 1], [[fn.Fneuron(2, w1), fn.Fneuron(2, w2)],[fn.Fneuron(2, w3)]])
    print('FAUX FAUX', mlp.aggregation(np.array([0, 0])))
    print('VRAI FAUX', mlp.aggregation(np.array([1, 0])))
    print('FAUX VRAI', mlp.aggregation(np.array([0, 1])))
    print('VRAI VRAI', mlp.aggregation(np.array([1, 1])))
