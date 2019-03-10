# Project 3
# Yubao Liu
# Nov 14, 2018

from functools import reduce
from exactInference import *
from xmlParser import *
import sys
import time


recursiveCallAmount = 0

# eliminationAsk and Class: Factor ****************************************************************************

def eliminationAsk(X, e, bn):
    """
    X: the query variable
    e: observed values for variables E
    bn: a Bayesian network specifying joint distribution P(X1, …, Xn)

    Compute bn's P(X|e) by variable elimination.

    >>> eliminationAsk('Burglary', dict(JohnCalls=T, MaryCalls=T), burglary)
    'False: 0.716, True: 0.284'

    """

    factors = []
    for var in reversed(bn.variables):  # Ordered
        factors.append(makeFactor(var, e, bn))
        if isHidden(var, X, e):     # if var is a hidden variable
            factors = sumOut(var, factors, bn)
    return pointwiseProduct(factors, bn).normalize()


def isHidden(var, X, e):
    """
    Is var a hidden variable or not?

    """
    return var != X and var not in e


def makeFactor(var, e, bn):
    """
    Return the factor for var in bn's joint distribution given e.
    # That is, bn's full joint distribution, projected to accord with e,
    # is the pointwise product of these factors for bn's variables.

    """
    node = bn.variableNode(var)
    variables = [X for X in [var] + node.parents if X not in e]
    cpt = {eventValues(e1, variables): node.pOfValue(e1[var], e1)
           for e1 in allEvents(variables, bn, e)}
    return Factor(variables, cpt)


def pointwiseProduct(factors, bn):
    global recursiveCallAmount
    recursiveCallAmount+=1
    return reduce(lambda f, g: f.pointwiseProduct(g, bn), factors)


def sumOut(var, factors, bn):
    """
    Eliminate var from all factors by summing over its values.

    """
    result, varFactors = [], []
    for f in factors:
        (varFactors if var in f.variables else result).append(f)
    result.append(pointwiseProduct(varFactors, bn).sumOut(var, bn))
    return result


class Factor:
    """
    A factor in a probability distribution.

    """

    def __init__(self, variables, cpt):
        self.variables = variables
        self.cpt = cpt

    def pointwiseProduct(self, other, bn):
        """
        Multiply two factors, combining their variables.

        """
        variables = list(set(self.variables) | set(other.variables))
        cpt = {eventValues(e, variables): self.p(e) * other.p(e)
               for e in allEvents(variables, bn, {})}
        return Factor(variables, cpt)

    def sumOut(self, var, bn):
        """
        New factor eliminating var by summing over its values.

        """
        variables = [X for X in self.variables if X != var]
        cpt = {eventValues(e, variables): sum(self.p(extend(e, var, val))
                                               for val in bn.variableValues(var))
               for e in allEvents(variables, bn, {})}
        return Factor(variables, cpt)

    def normalize(self):
        """
        Returns the normalized probabilities

        """
        # assert len(self.variables) == 1
        return ProbDist(self.variables[0],
                        {k: v for ((k,), v) in self.cpt.items()})

    def p(self, e):
        """
        Look up for e.

        """
        return self.cpt[eventValues(e, self.variables)]


def allEvents(variables, bn, e):
    """
    Yield all the ways of extending e with values for all variables.

    """
    if not variables:
        yield e
    else:
        X, rest = variables[0], variables[1:]
        for e1 in allEvents(rest, bn, e):
            for x in bn.variableValues(X):
                yield extend(e1, X, x)

# main functions ****************************************************************************
if __name__ == '__main__':
    xmlDir = sys.argv[1]
    queryVar = sys.argv[2]
    evidenceList = sys.argv[3:]
    evidenceKeyList = [evidenceList[i] for i in range(len(evidenceList)) if i%2 == 0]
    evidenceValueList = [evidenceList[i] for i in range(len(evidenceList)) if i%2 == 1]
    evidenceDict = {}
    for i in range(len(evidenceKeyList)):
        evidenceDict[evidenceKeyList[i]] = bool(evidenceValueList[i])

    xmlBayesNet = bayesNetFromXML(xmlDir)
    startTime = time.time()
    result = eliminationAsk(queryVar, evidenceDict, xmlBayesNet)
    print((result[T],result[F]))
    print("P(True):", result[T])
    print("P(False):", result[F])
    print("the amount of recursion called: ", recursiveCallAmount)
    print("the time consumed to caculate: ", time.time()- startTime)


