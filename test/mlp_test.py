#! /usr/bin/env python
#-*- coding: utf-8 -*-

"""
author  : Adrien Lafage\n
date    : 04.19
Multi-layer perceptron Unitary Tests
=============
Provides
    - Unitary tests for multi-layer perceptron

Documentation`_
-------------
...
"""

############| IMPORTS |#############
import os
import sys
sys.path.append('../src')

import numpy as np
import unittest as unittest
import fneuron as fn
import mlp as mlp
####################################


#############| NOTES |##############
""" 
"""
####################################

####################################
############| CLASSES |#############
####################################

class Mlp_test(unittest.TestCase):
    """ Unitary tests for multi-layer perceptron """
    def testXOR_computes_sigmoid(self):
        """ Computing function test
        Mlp as a XOR logic
        """
        # weight initialization
        w1 = np.array([-1, 2, 2]) #OR logic
        w2 = np.array([3, -2, -2]) # NAND logic
        w3 = np.array([-3, 2, 2]) # AND logic
        # XOR logic mlp
        multiLayerPerceptron = mlp.Mlp(2, 1, [2, 1], [[fn.Fneuron(2, w1), fn.Fneuron(2, w2)], [fn.Fneuron(2, w3)]])
        # first test
        output = multiLayerPerceptron.compute(np.array([0, 0]), fn.Fneuron.sigmoid)
        expected = np.array([float(0)])
        self.assertEqual(output, expected)
        # second test
        output = multiLayerPerceptron.compute(np.array([0, 1]), fn.Fneuron.sigmoid)
        expected = np.array([float(1)])
        self.assertEqual(output, expected)
        # third test
        output = multiLayerPerceptron.compute(np.array([1, 0]), fn.Fneuron.sigmoid)
        expected = np.array([float(1)])
        self.assertEqual(output, expected)
        # fourth test
        output = multiLayerPerceptron.compute(np.array([1, 1]), fn.Fneuron.sigmoid)
        expected = np.array([float(0)])
        self.assertEqual(output, expected)
    
    def testAND_computes_sigmoid(self):
        """ Computing function test
        Mlp as an AND logic
        """
        # weight initialization
        w1 = np.array([-3, 2, 2])
        # AND logic mlp
        multiLayerPerceptron = mlp.Mlp(2, 1, [1], [[fn.Fneuron(2, w1)]])
        # first test
        output = multiLayerPerceptron.compute(np.array([0, 0]), fn.Fneuron.sigmoid)
        expected = np.array([float(0)])
        self.assertEqual(output, expected)
        # second test
        output = multiLayerPerceptron.compute(np.array([1, 0]), fn.Fneuron.sigmoid)
        expected = np.array([float(0)])
        self.assertEqual(output, expected)
        # third test
        output = multiLayerPerceptron.compute(np.array([0, 1]), fn.Fneuron.sigmoid)
        expected = np.array([float(0)])
        self.assertEqual(output, expected)
        # fourth test 
        output = multiLayerPerceptron.compute(np.array([1, 1]), fn.Fneuron.sigmoid)
        expected = np.array([float(1)])
        self.assertEqual(output, expected)
    
    def testOR_computes_sigmoid(self):
        """ Computing function test
        Mlp as an OR logic
        """
        # multi-layer function init
        w1 = np.array([-1, 2, 2])
        # OR logic mlp
        multiLayerPerceptron = mlp.Mlp(2, 1, [1], [[fn.Fneuron(2, w1)]])
        # first test
        output = multiLayerPerceptron.compute(np.array([0, 0]), fn.Fneuron.sigmoid)
        expected = np.array([float(0)])
        self.assertEqual(output, expected)
        # second test
        output = multiLayerPerceptron.compute(np.array([1, 0]), fn.Fneuron.sigmoid)
        expected = np.array([float(1)])
        self.assertEqual(output, expected)
        # third test
        output = multiLayerPerceptron.compute(np.array([0, 1]), fn.Fneuron.sigmoid)
        expected = np.array([float(1)])
        self.assertEqual(output, expected)
        # fourth test 
        output = multiLayerPerceptron.compute(np.array([1, 1]), fn.Fneuron.sigmoid)
        expected = np.array([float(1)])
        self.assertEqual(output, expected)




####################################
############| PROGRAM |#############
####################################

if __name__ == "__main__":
    # run all the tests
    unittest.main()