# Project 3
# Yubao Liu
# Nov 14, 2018

import random

# BayesNet and BayesNode ****************************************************************************

class BayesNet:
    """
    Bayesian network structure

    """

    def __init__(self, nodeInfos=None):
        """
        Nodes must be ordered with parents before children.

        """
        self.nodes = []
        self.variables = []
        nodeInfos = nodeInfos or []
        for nodeInfo in nodeInfos:
            self.add(nodeInfo)

    def add(self, nodeInfo):
        """
        Add a node to the net. 
        # Its parents must already be in the net, and its variable must not.

        """
        node = BayesNode(*nodeInfo)
        # assert node.variable not in self.variables
        # assert all((parent in self.variables) for parent in node.parents)
        self.nodes.append(node)
        self.variables.append(node.variable)
        for parent in node.parents:
            self.variableNode(parent).children.append(node)

    def variableNode(self, var):
        """
        Return the node for the variable named var.
        >>> burglary.variableNode('Burglary').variable
        'Burglary'

        """
        for n in self.nodes:
            if n.variable == var:
                return n
        raise Exception("No such variable: {}".format(var))

    def variableValues(self, var):
        """
        Return the domain of var.

        """
        return [True, False]

    def __repr__(self):
        return 'BayesNet({0!r})'.format(self.nodes)

class BayesNode:
    """
    The node structure of Bayesian Network
    P(X | parents). 

    """

    def __init__(self, X, parents, cpt):
        """
        X:          a variable name
        parents:    a sequence of variable names or a space-separated string.  
        cpt:        the conditional probability table
        
        Examples:
        >>> X = BayesNode('B', '', 0.001)
        >>> Y = BayesNode('J', 'A', {T: 0.9, F: 0.05})
        >>> Z = BayesNode('A', 'B E',
        ...    {(T, T): 0.95, (T, F): 0.94, (F, T): 0.29, (F, F): 0.001})

        """
        # parents: sometimes more than one parent
        if isinstance(parents, str):
            parents = parents.split()

        # cpt
        if isinstance(cpt, (float, int)):  # no parents
            cpt = {(): cpt}
        elif isinstance(cpt, dict): # one parent
            if cpt and isinstance(list(cpt.keys())[0], bool):
                cpt = {(v,): p for v, p in cpt.items()}

        # components of the BayesNode
        self.variable = X
        self.parents = parents
        self.cpt = cpt
        self.children = []

    def pOfValue(self, value, event):
        """
        Returns the conditional probability of X equals to value when parents' value equal to event
        Examples:
        >>> bn = BayesNode('X', 'B', {T: 0.2, F: 0.625})
        >>> bn.pOfValue(False, {'B': False, 'E': True})
        0.375

        """
        assert isinstance(value, bool)
        pTrue = self.cpt[eventValues(event, self.parents)]
        return pTrue if value else 1 - pTrue

    def sample(self, event):
        """
        Returns True/False randomly according to the conditional probability

        """
        return self.pOfValue(True, event) > random.uniform(0.0, 1.0)


    def __repr__(self):
        return repr((self.variable, ' | '.join(self.parents)))



# util functions ****************************************************************************

def eventValues(event, variables):
    """
    Returns values in event
    >>> eventValues ({'A': 0.1, 'B': 0.9, 'X': 0.8}, ['X', 'A'])
    (0.8, 0.1)
    >>> eventValues ((0.1, 0.2), ['C', 'A'])
    (0.1, 0.2)
    
    """
    if isinstance(event, tuple) and len(event) == len(variables):
        return event
    else:
        return tuple([event[var] for var in variables])

