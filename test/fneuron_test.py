#! /usr/bin/env python
#-*- coding: utf-8 -*-

"""
author  : Adrien Lafage\n
date    : 04.19
Formal Neuron Unitary Tests
=============
Provides
    - Unitary tests for Formal neuron

Documentation

You can find it here <https://github.com/Aydens01/neural-networks/blob/dev/doc/fneuron.md>`_
-------------
...
"""

############| IMPORTS |#############
import os
import sys
sys.path.append('../src')

import numpy as np
import unittest
import fneuron as fn
####################################


#############| NOTES |##############
""" 
"""
####################################

####################################
############| CLASSES |#############
####################################

class Fneuron_test(unittest.TestCase):
    """Unitary tests for formal neuron"""
    def test_heaviside(self):
        """ Heaviside activation function test 
        """
        # formal neuron init
        neuron = fn.Fneuron(2, np.array([10, 1, -1]))
        # first test
        output = neuron.heaviside(np.array([100, 0]))
        expected = 1.0
        self.assertEqual(output, expected)
        # second test
        output = neuron.heaviside(np.array([0, 100]))
        expected = 0.0
        self.assertEqual(output, expected)
        # third test
        output = neuron.heaviside(np.array([0,0]))
        expected = 1.0
        self.assertEqual(output, expected)
        # fourth test
        output = neuron.heaviside(np.array([90, 100]))
        expected = 1.0
        self.assertEqual(output, expected)
        #fifth test
        output = neuron.heaviside(np.array([0, 10]))
        expected = 1.0
        self.assertEqual(output, expected)

    def test_sigmoid(self):
        """ Sigmoid activation function test
        """
        # formal neuron init
        neuron = fn.Fneuron(2, np.array([10,1,-1]))
        # first test
        output = neuron.sigmoid(np.array([100, 0]), 1000)
        expected = 1.0
        self.assertEqual(output, expected)
        # second test
        output = neuron.sigmoid(np.array([0, 100]), 1000)
        expected = 0.0
        self.assertEqual(output, expected)
        # third test
        output = neuron.sigmoid(np.array([0,0]), 1000)
        expected = 1.0
        self.assertEqual(output, expected)
        # fourth test
        output = neuron.sigmoid(np.array([90, 100]), 1000)
        expected = 0.5
        self.assertEqual(output, expected)
        #fifth test
        output = neuron.sigmoid(np.array([0, 10]), 1000)
        expected = 0.5
        self.assertEqual(output, expected)


####################################
############| PROGRAM |#############
####################################

if __name__ == "__main__":
    # run all the tests
    unittest.main()

