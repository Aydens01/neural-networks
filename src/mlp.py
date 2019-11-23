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
import perceptron as pct
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
    def __init__(self, size_in, size_out, layers_sizes, layers = None):
        """ Initialize the multi-layer perceptron
        Parameters :
        ------------
            size_in {int} : 
            size_out {int} :
            layers_sizes : 
        """
        self.size_in = size_in
        self.size_out = size_out
        self.layers_sizes = layers_sizes
        self.layers = layers
    
    ##################################
    # Some decision functions associated 
    # with the Mlp object
    ##################################
    def compute(self, inputs, decision_fct=fn.Fneuron.sigmoid):
        """ Computing function of the mpl
        using formal neurons
        Parameters :
        ------------
            inputs {np.array} : mlp inputs
        Output :
        ------------
            outputs {np.array}
        """
        # set up
        network_output = None
        tmp = inputs

        for layer in self.layers :
            layer_output = np.array([])
            for neuron in layer :
                # computing with the chosen decision function
                layer_output = np.append(layer_output, decision_fct(neuron, tmp))
            tmp = layer_output
            network_output = layer_output
        return(network_output)
    
    # FIXME: use perceptron in progress
    def dev(self, inputs, decision_fct=pct.Perceptron.softmax):
        """ Computing function of the mlp
        using perceptrons
        Parameters :
        ------------
            inputs {np.array} : mlp inputs
        Output :
        --------
            outputs {np.array}
        """
        # set up
        network_output = None
        tmp = inputs

        for layer in self.layers:
            layer_output = None
            # TODO: init perceptrons
            tmp = layer_output
            network_output = layer_output
        
        return(network_output)


####################################
############| PROGRAM |#############
####################################

if __name__=="__main__":

    #>>> DEV TESTS <<<#
    pass