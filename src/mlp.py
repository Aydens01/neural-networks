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
    def __init__(self, size_in, size_out, layers_sizes=[], layers=[np.array()]):
        self.size_in = size_in
        self.size_out = size_out
        self.layers_sizes = layers_sizes
        self.layers = layers
    
    ##################################
    # Some decision functions provides 
    # with the Mlp object
    ##################################
    def heaviside(self, responses):
        """ Heaviside activation function
        """
        outputs = [(1 if response>0 else 0) for response in responses]
        return outputs
    

    def classification(self, inputs, decision_fct=heaviside):
        """ Aggregation function of the perceptron
        """
        outputs = []
        for layer in self.layers:
            outputs.append(np.dot(inputs, layer))
        return outputs


####################################
############| PROGRAM |#############
####################################

if __name__=="__main__":

    #>>> DEV TESTS <<<#
    test = Mlp(2, 1, [3], [np.array([[-10, 1, 0],[45, -1, -1],[-10, 0, 1]])])