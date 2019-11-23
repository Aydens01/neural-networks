#! /usr/bin/env python
#-*- coding: utf-8 -*-

"""
author  : Adrien Lafage\n
date    : 03.19
Formal neuron
=============
Provides
-------------
    - Initialization of the formal neuron
    - Aggregation function
    - Some decision functions : Heaviside and Sigmoid

Using
-------------
```py
# creation
neuron = Fneuron(size, weights)
# use heaviside decision function
neuron.heaviside(inputs)
# use sigmoid decision function
neuron.sigmoid(inputs, a)
```

Documentation
-------------
You can find it here <https://github.com/Aydens01/neural-networks/blob/dev/doc/fneuron.md>
"""

############| IMPORTS |#############
import os
import sys
import numpy as np
####################################


#############| NOTES |##############
"""
OK
"""
####################################


####################################
############| CLASSES |#############
####################################

class Fneuron():
    "Formal neuron"
    def __init__(self, size, weights=np.array([])):
        """ Construct function of the formal neuron

        Parameters :
        ------------
            size {int} : number of inputs\n
            weights {np.array} : array of the formal neuron weights
        """
        self.weights = np.zeros(size+1) if len(weights)!=size+1 else weights
    
    def aggregation(self, inputs):
        """ Aggregation function of the formal neuron

        Parameters :
        ------------
            inputs {np.array} : formal neuron inputs
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
            output {float} : result of the neuron with these inputs and this decision function.
        """
        return(1.0 if self.aggregation(inputs)>=0 else 0.0)
    
    def sigmoid(self, inputs, a=1000):
        """ Sigmoid activation function\n
        Probabilistic interpretation.
        Parameters :
        ------------
            inputs {np.array} : formal neuron 
            a {int} : tilt coefficient
        Output :
        --------
            output {int} : result of the neuron with these inputs and this decision function
        """
        return(1/(1+np.exp(-a*self.aggregation(inputs))))

    def __repr__(self):
        """ Formal neuron status
        Output :
        --------
            status {str}
        """
        status = "_______________________\n"
        status += "     Formal Neuron     \n\n"
        for i in range(len(self.weights)):
            status += "w"+str(i)+" : "+str(self.weights[i])+"\n"
        status += "_______________________\n"
        return(status) 

####################################
############| PROGRAM |#############
####################################

if __name__=="__main__":

    #>>> DEV TESTS <<<#
    pass