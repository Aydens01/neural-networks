#! /usr/bin/env python
#-*- coding: utf-8 -*-

"""
author  : Adrien Lafage\n
date    : 03.19
Perceptrons Tests Unitaires
=============
Provides


Documentation
-------------
...
"""

############| IMPORTS |#############
import os
import sys
sys.path.append('../src')

import numpy as np
import unittest
import perceptrons as pct
####################################


#############| NOTES |##############
"""
TODO:
"""
####################################

####################################
############| CLASSES |#############
####################################

class Percep_test(unittest.TestCase):
    """Unitary tests for perceptron"""
    def test_or_2d(self):
        """ 2D : perceptron's learning to be an OR function
        """
        # Training sample definition
        training_data = [np.array([0, 0]),
                         np.array([0, 1]),
                         np.array([1, 0]), 
                         np.array([1, 1])]
        labels = [[0], [1], [1], [1]]
        # Perceptron initialization
        percep = pct.Perceptron(2)
        # Perceptron training
        percep.train(training_data, labels)
        # Testing sample definition
        testing_data = [np.array([1, 1]), 
                        np.array([0, 0]), 
                        np.array([0, 0]), 
                        np.array([0, 1]), 
                        np.array([1, 0]), 
                        np.array([1, 1]), 
                        np.array([0, 0])]
        # Output given by the perceptron for the testing sample
        output = [percep.activation(data) for data in testing_data]
        # Output expected
        expected = [[1], [0], [0], [1], [1], [1], [0]]
        # Equality test
        self.assertEqual(output, expected)
    
    def test_or_3d(self):
        """ 3D : perceptron's learning to be an OR function
        """
        # Training sample definition
        training_data = [np.array([0, 0, 0]),
                         np.array([0, 0, 1]),
                         np.array([0, 1, 0]),
                         np.array([1, 0, 0]),
                         np.array([1, 0, 1]),
                         np.array([1, 1, 0]),
                         np.array([0, 1, 1]),
                         np.array([1, 1, 1])]
        labels = [[0], [1], [1], [1], [1], [1], [1], [1]]
        # Perceptron initialization
        percep = pct.Perceptron(3)
        # Perceptron training
        percep.train(training_data, labels)
        # Testing sample definition
        testing_data = [np.array([1, 1, 0]), 
                        np.array([0, 0, 0]), 
                        np.array([0, 0, 0]), 
                        np.array([0, 1, 0]), 
                        np.array([1, 0, 0]), 
                        np.array([1, 1, 0]), 
                        np.array([0, 0, 0])]
        # Output given by the perceptron for the testing sample
        output = [percep.activation(data) for data in testing_data]
        # Output expected
        expected = [[1], [0], [0], [1], [1], [1], [0]]
        # Equality test
        self.assertEqual(output, expected)

    def test_and_2d(self):
        """ 2D : perceptron's learning to be an AND function
        """
        # Training sample definition
        training_data = [np.array([0, 0]),
                         np.array([0, 1]),
                         np.array([1, 0]), 
                         np.array([1, 1])]
        labels = [[0], [0], [0], [1]]
        # Perceptron initialization
        percep = pct.Perceptron(2)
        # Perceptron training
        percep.train(training_data, labels)
        # Testing sample definition
        testing_data = [np.array([1, 1]), 
                        np.array([0, 0]), 
                        np.array([0, 0]), 
                        np.array([0, 1]), 
                        np.array([1, 0]), 
                        np.array([1, 1]), 
                        np.array([0, 0])]
        # Output given by the perceptron for the testing sample
        output = [percep.activation(data) for data in testing_data]
        # Output expected
        expected = [[1], [0], [0], [0], [0], [1], [0]]
        # Equality test
        self.assertEqual(output, expected)     

    def test_and_3d(self):
        """ 3D : perceptron's learning to be an AND function
        """
        # Training sample definition
        training_data = [np.array([0, 0, 0]),
                         np.array([0, 0, 1]),
                         np.array([0, 1, 0]),
                         np.array([1, 0, 0]),
                         np.array([1, 0, 1]),
                         np.array([1, 1, 0]),
                         np.array([0, 1, 1]),
                         np.array([1, 1, 1])]
        labels = [[0], [0], [0], [0], [0], [0], [0], [1]]
        # Perceptron initialization
        percep = pct.Perceptron(3)
        # Perceptron training
        percep.train(training_data, labels)
        # Testing sample definition
        testing_data = [np.array([1, 1, 0]), 
                        np.array([0, 0, 0]), 
                        np.array([0, 0, 0]), 
                        np.array([0, 1, 0]), 
                        np.array([1, 0, 0]), 
                        np.array([1, 1, 1]), 
                        np.array([0, 0, 0])]
        # Output given by the perceptron for the testing sample
        output = [percep.activation(data) for data in testing_data]
        # Output expected
        expected = [[0], [0], [0], [0], [0], [1], [0]]
        # Equality test
        self.assertEqual(output, expected)

####################################
############| PROGRAM |#############
####################################

if __name__ == "__main__":
    # run all the tests
    unittest.main()