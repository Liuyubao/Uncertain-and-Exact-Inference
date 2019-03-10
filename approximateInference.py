# Project 3
# Yubao Liu
# Nov 14, 2018

from probDistribution import *
from bayesNet import *
from xmlParser import *
import sys
import time


T, F = True, False

# Prior sampling ***************************************************************************

def priorSample(bn):
    """
    bn: a Bayesian network

    Randomly sample from bn's distribution. 
    Returns a {variable: value} dict.

    """
    event = {}
    for node in bn.nodes:
        event[node.variable] = node.sample(event)
    return event


# Rejection sampling ****************************************************************************

def rejectionSampling(X, e, bn, N=10000):
    """
    X: the query variable
    e: observed values for variables E
    bn: a Bayesian network
    N: the total number of samples to be generated

    Returns the probability distribution of variable X given e in BayesNet bn, using N samples.
    
    Examples:
    # >>> random.seed(47)
    >>> rejectionSampling('Burglary', dict(JohnCalls=T, MaryCalls=T), burglary, 10000)
    'False: 0.7, True: 0.3'
    
    """
    counts = {x: 0 for x in bn.variableValues(X)}  # the dict to count the number of each variable's value
    for j in range(N):
        sample = priorSample(bn) # the sample dict from the distribution
        if consistentWith(sample, e):
            counts[sample[X]] += 1
    return ProbDist(X, counts)


def consistentWith(event, evidence):
    """
    Returns if event consistent with the given evidence?

    """
    for k,v in evidence.items():
        if event[k] != v:
            return False
    return True

# likelihood weighting sampling ****************************************************************************

def likelihoodWeighting(X, e, bn, N=10000):
    """
    X: the query variable
    e: observed values for variables E
    bn: a Bayesian network specifying joint distribution P(X1, …, Xn)
    N: the total number of samples to be generated

    Returns the probability distribution of variable X given e in BayesNet bn.  
    # >>> random.seed(1017)
    >>> likelihoodWeighting('Burglary', dict(JohnCalls=T, MaryCalls=T), burglary, 10000)
    'False: 0.702, True: 0.298'
    
    """
    W = {x: 0 for x in bn.variableValues(X)}   # the dict to count the number of each variable's value
    for j in range(N):
        sample, weight = weightedSample(bn, e)  # boldface x, w in [Figure 14.15]
        W[sample[X]] += weight
    return ProbDist(X, W)

def weightedSample(bn, e):
    """
    Returns the event and its weight, the likelihood that the event
    accords to the evidence.
    """
    w = 1
    event = dict(e)  # the event initial with evidence value
    for node in bn.nodes:
        Xi = node.variable
        if Xi in e:
            w *= node.pOfValue(e[Xi], event)     # w ← w × P(Xi = xi | parents(Xi))
        else:
            event[Xi] = node.sample(event)      # x[i] ← a random sample from P(Xi | parents(Xi))
    return event, w

# Gibbs sampling ****************************************************************************


def gibbsAsk(X, e, bn, N=1000):
    """
    X: the query variable
    e: observed values for variables E
    bn: a Bayesian network specifying joint distribution P(X1, …, Xn)
    N: the total number of samples to be generated

    """

    counts = {x: 0 for x in bn.variableValues(X)}  # the dict to count the number of each variable's value
    Z = [var for var in bn.variables if var not in e]   # Z: the nonevidence variables in bn
    xState = dict(e) # the current state of the network, initially copied from e
    for Zi in Z:
        xState[Zi] = random.choice(bn.variableValues(Zi))
    for j in range(N):
        for Zi in Z:
            xState[Zi] = markovBlanketSample(Zi, xState, bn)  # set the value of Zi in xState by sampling from P(Zi | mb(Zi))
            counts[xState[X]] += 1
    return ProbDist(X, counts)


def markovBlanketSample(X, e, bn):
    """
    Return a sample from P(X | mb) where mb denotes that the
    variables in the Markov blanket of X take their values from event
    e (which must assign a value to each). The Markov blanket of X is
    X's parents, children, and children's parents.

    """
    Xnode = bn.variableNode(X)
    Q = ProbDist(X)
    for xi in bn.variableValues(X):
        ei = extend(e, X, xi)
        # [Equation 14.12:]
        Q[xi] = Xnode.pOfValue(xi, e) * product(Yj.pOfValue(ei[Yj.variable], ei)
                                         for Yj in Xnode.children)
    # (assuming a Boolean variable here)
    return Q.normalize()[True] > random.uniform(0.0, 1.0)

# util functions ****************************************************************************

def product(numbers):
    """
    Return the product of the numbers
    Examples:
    product([2, 3, 10]) == 60

    """
    result = 1
    for x in numbers:
        result *= x
    return result

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
    algChoice = int(sys.argv[1])
    sampleNum = int(sys.argv[2])
    xmlDir = sys.argv[3]
    queryVar = sys.argv[4]
    evidenceList = sys.argv[5:]
    evidenceKeyList = [evidenceList[i] for i in range(len(evidenceList)) if i%2 == 0]
    evidenceValueList = [evidenceList[i] for i in range(len(evidenceList)) if i%2 == 1]
    evidenceDict = {}
    for i in range(len(evidenceKeyList)):
        evidenceDict[evidenceKeyList[i]] = bool(evidenceValueList[i])

    xmlBayesNet = bayesNetFromXML(xmlDir)
    startTime = time.time()
    if algChoice == 1:
        result = rejectionSampling(queryVar, evidenceDict, xmlBayesNet, sampleNum)
    elif algChoice == 2:
        result = likelihoodWeighting(queryVar, evidenceDict, xmlBayesNet, sampleNum)
    elif algChoice == 3:
        result = gibbsAsk(queryVar, evidenceDict, xmlBayesNet, sampleNum)
    print((result[T],result[F]))
    print("P(True):", result[T])
    print("P(False):", result[F])
    print("the time consumed to caculate: ", time.time()- startTime)



