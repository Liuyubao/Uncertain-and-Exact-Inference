# Project 3
# Yubao Liu
# Nov 14, 2018

from bayesNet import *
from probDistribution import *
from xmlParser import *
import sys
import time


# global recursiveCallAmount
recursiveCallAmount = 0

# enumerationAsk and enumerationAll****************************************************************************


def enumerationAsk(X, e, bn):
    """
    X: the query variable
    e: observed values for variables E
    bn: a BayesNet with variables {X} ⋃ E ⋃ Y /* Y = hidden variables */

    Return the conditional probability distribution of variable X given evidence e, from BayesNet bn. 
    Examples:
    >>> enumerationAsk('Burglary', dict(JohnCalls=T, MaryCalls=T), burglary)
    'False: 0.716, True: 0.284'

    """
    Q = ProbDist(X)     # a distribution over X, initially empty
    for xi in bn.variableValues(X):
        Q[xi] = enumerateAll(bn.variables, extend(e, X, xi), bn)
    return Q.normalize()


def enumerateAll(variables, e, bn):
    """
    Returns the sum of those entries in P(variables | e{others})
    consistent with e, where P is the joint distribution represented
    by bn
    
    Parents must precede children in variables.
    """
    global recursiveCallAmount
    if not variables:       # if EMPTY?(vars) then return 1.0
        return 1.0
    Y, rest = variables[0], variables[1:]   # Y ← FIRST(vars)
    Ynode = bn.variableNode(Y)
    if Y in e:
        recursiveCallAmount+=1
        return Ynode.pOfValue(e[Y], e) * enumerateAll(rest, e, bn)
    else:
        # global recursiveCallAmount
        recursiveCallAmount+=1
        return sum(Ynode.pOfValue(y, e) * enumerateAll(rest, extend(e, Y, y), bn)
                   for y in bn.variableValues(Y))


# util functions ****************************************************************************

# Functions on Sequences and Iterables
def extend(s, var, val):
    """
    Copy the substitution s and extend it by setting var to val; return copy.
    >>> extend({x: 1}, y, 2) == {x: 1, y: 2}
    True
    
    """
    s2 = s.copy()
    s2[var] = val
    return s2

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
    result = enumerationAsk(queryVar, evidenceDict, xmlBayesNet)
    print((result[T],result[F]))
    print("P(True):", result[T])
    print("P(False):", result[F])
    print("the amount of recursion called: ", recursiveCallAmount)
    print("the time consumed to caculate: ", time.time()- startTime)


