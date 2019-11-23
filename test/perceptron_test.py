#! /usr/bin/env python
#-*- coding: utf-8 -*-

"""
author  : Adrien Lafage\n
date    : 05.19
Perceptron Unitary Tests
=============
Provides
    - Unitary tests for Perceptron

Documentation

You can find it here <https://github.com/Aydens01/neural-networks/blob/dev/doc/perceptron.md>`_
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
import perceptron as pct
####################################

#############| NOTES |##############
"""
todo: write some tests 
"""
####################################

####################################
############| CLASSES |#############
####################################

class Perceptron_test(unittest.TestCase):
    """Unitary tests for perceptron"""
    # ? what test could be relevant?
    def testAND_heaviside(self):
        """ Perceptron as an AND logic
        with heaviside function
        """
        # weight initialization
        w1 = np.array([-3, 2, 2])
        # AND logic perceptron
        perceptron = pct.Perceptron(2, 1, [fn.Fneuron(2, w1)])
        output = perceptron.heaviside(np.array([0, 0]))
        expected = np.array([0.0])
        self.assertEqual(output, expected)
        # second test
        output = perceptron.heaviside(np.array([1, 0]))
        expected = np.array([0.0])
        self.assertEqual(output, expected)
        # third test
        output = perceptron.heaviside(np.array([0, 1]))
        expected = np.array([0.0])
        self.assertEqual(output, expected)
        # fourth test
        output = perceptron.heaviside(np.array([1, 1]))
        expected = np.array([1.0])
        self.assertEqual(output, expected)
    
    def testOR_heaviside(self):
        """ Perceptron as an OR logic
        with heaviside function
        """
        # weight initialization
        w1 = np.array([-1, 2, 2])
        # OR logic fneuron
        perceptron = pct.Perceptron(2, 1, [fn.Fneuron(2, w1)])
        # first test
        output = perceptron.heaviside(np.array([0, 0]))
        expected = np.array([0.0])
        self.assertEqual(output, expected)
        # second test
        output = perceptron.heaviside(np.array([1, 0]))
        expected = np.array([1.0])
        self.assertEqual(output, expected)
        # third test
        output = perceptron.heaviside(np.array([0, 1]))
        expected = np.array([1.0])
        self.assertEqual(output, expected)
        # fourth test
        output = perceptron.heaviside(np.array([1, 1]))
        expected = np.array([1.0])
        self.assertEqual(output, expected)
    
    def testAND_sigmoid(self):
        """ Perceptron as an AND logic
        with sigmoid function
        """
        # weight initialization
        w1 = np.array([-1, 2, 2])
        # OR logic fneuron
        perceptron = pct.Perceptron(2, 1, [fn.Fneuron(2, w1)])
        pass


####################################
############| PROGRAM |#############
####################################

if __name__ == "__main__":
    # run all the tests
    unittest.main()