#! /usr/bin/env python
#-*- coding: utf-8 -*-

"""
author  : Adrien Lafage\n
date    : 25.03
Views
=============
Provides
    1. A standard data object
    2. Print functions for data

Documentation
-------------
...
"""

############| IMPORTS |#############
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
####################################


#############| NOTES |##############
"""
TODO: - write all the annotations
"""
####################################


####################################
############| CLASSES |#############
####################################

class Data2D():
    """ Data class """
    def __init__(self, value, label=None):
        """ Construct function of Data2D

        Parameters:
        ------------
            value {list} : TODO:
            label {str} : TODO:
        """
        self.value = value
        self.label = label

    def __print__(self):
        """ Define the print function for Data2D
        """
        print("value : {}, label : {}".format(self.value, self.label))


class DataSet2D():
    """ DataSet2D class"""
    def __init__(self, fct, mini=0, maxi=1, size=500):
        """ Conctruct function of DataSet2D

        Parameters:
        ------------
        """
        self.mini = mini
        self.maxi = maxi
        self.size = size
        self.fct  = fct
        self.data = self.generate()
    
    def generate(self):
        data = []
        for i in range(500):
            value = (np.random.randint(self.mini, self.maxi, size=(1, 2)))[0]
            data.append(Data2D(value, self.fct(value)))
        return(data)

class View():
    """ Data view """
    def __init__(self, sample, xlabel="x", ylabel="y"):
        """ Construct function of the data view
        
        Parameters :
        ------------
            data {dict} : dictionnary with the labels and 
            the sample.\n
        """
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.sample = sample

    def __print__(self):
        cmap = 'brcmykwg' #FIXME: allows only 9 colors

        for d in self.sample:
            plt.scatter(d.value[0], d.value[1], s=None, c=cmap[int(d.label)])
        
        plt.xlabel(self.xlabel+" axis")
        plt.ylabel(self.ylabel+" axis")
        plt.show()


if __name__=="__main__":
    pass