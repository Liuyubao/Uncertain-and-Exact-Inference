# Project 3
# Yubao Liu
# Nov 14, 2018

from bayesNet import *
from math import isclose

class ProbDist:
    """
    A discrete probability distribution.
    >>> P = ProbDist('Flip'); P['H'], P['T'] = 0.25, 0.75; 
    >>> P['H']
    0.25
    >>> P = ProbDist('X', {'a': 125, 'b': 375, 'c': 500})
    >>> P['a'], P['b'], P['c']
    (0.125, 0.375, 0.5)
    
    """

    def __init__(self, varName='*', freqs=None):
        """
        Components:
        prob: the probability of each variables' value
        varName: the name of the variable
        values: the values of the variable
        If freqs is given, make it normalized.

        """
        self.prob = {}
        self.varName = varName
        self.values = []
        if freqs != None:
            for (v, p) in freqs.items():
                self[v] = p
            self.normalize()

    def __getitem__(self, val):
        """
        Given a value using [], return P(value).

        """

        return self.prob[val]

    def __setitem__(self, val, p):
        """
        Set P(val) = p using [].

        """
        if val not in self.values:
            self.values.append(val)
        self.prob[val] = p

    def normalize(self):
        """
        All values sum to 1.

        """
        total = sum(self.prob.values())
        if not isclose(total, 1.0):
            for val in self.prob:
                self.prob[val] /= total
        return self


    def __repr__(self):
        return "P({})".format(self.varName)