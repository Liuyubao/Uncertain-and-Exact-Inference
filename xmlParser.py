# Project 3
# Yubao Liu
# Nov 14, 2018

from xml.dom.minidom import parse
import xml.dom.minidom
from bayesNet import *

# XML Parser ****************************************************************************
def bayesNetFromXML(xmlDir):
    """
    xmlDir: the dir of the xml file

    returns a BayesNet based on the xml file

    """

    # used minidom to open the xml
    DOMTree = xml.dom.minidom.parse(xmlDir)
    bif = DOMTree.documentElement   # the root element
    network = bif.getElementsByTagName("NETWORK")[0]
    # varibles = network.getElementsByTagName("VARIABLE")
    nodeInfos = network.getElementsByTagName("DEFINITION")
    # amountOfNodes = len(nodeInfos) # store the amount of the nodes


    # rearrange the node list to make sure it ordered with parents before children.
    nodeInfoStack = []
    # first: put nodes with no parents into the stack
    for nodeInfo in nodeInfos:
        if len(nodeInfo.getElementsByTagName("GIVEN")) == 0:
            varName = nodeInfo.getElementsByTagName("FOR")[0].childNodes[0].data
            nodeInfoStack.append([varName, nodeInfo])
            nodeInfos.remove(nodeInfo)

    while len(nodeInfos) > 0:
        varNameList = [node[0] for node in nodeInfoStack]
        for nodeInfo in nodeInfos:
            varName = nodeInfo.getElementsByTagName("FOR")[0].childNodes[0].data
            allInVarNameList = True
            for givenNode in nodeInfo.getElementsByTagName("GIVEN"):
                if not givenNode.childNodes[0].data in varNameList:
                    allInVarNameList = False
                    break
            if allInVarNameList:
                nodeInfoStack.append([varName, nodeInfo])
                nodeInfos.remove(nodeInfo)

    nodeInfos = [node[1] for node in nodeInfoStack]



    nodeTupleList = []  # used to give to Bayesnet input parameter

    for nodeInfo in nodeInfos:
        # 1. var Name, String
        varName = nodeInfo.getElementsByTagName("FOR")[0].childNodes[0].data
        # 2. parents Name, string like 'Burglary Earthquake'
        parentsName = ""    
        for parent in nodeInfo.getElementsByTagName("GIVEN"):
            parentsName += (" " + parent.childNodes[0].data)
        parentsName = parentsName.strip()

        cptValuesStr = ""  # all the cpt values in attribute TABLE
        cptNode = nodeInfo.getElementsByTagName("TABLE")[0]
        for item in cptNode.childNodes:
            if type(item) is xml.dom.minidom.Text:
                cptValuesStr += (" " + item.data.strip())

        # replace all \n and \t by " "
        cptValuesStr.replace("\n", " ")
        cptValuesStr.replace("\t", " ")

        # save all the cpt values to a list
        cptValuesList = cptValuesStr.strip().split(" ")
        while " " in cptValuesList:
            cptValuesList.remove(" ")
        while "" in cptValuesList:
            cptValuesList.remove("")

        # only leave the odd index, which is the cpt evaluated to True
        cptValuesList = [cptValuesList[i] for i in range(len(cptValuesList)) if i % 2 == 0]
        
        # print("*****cptValuesStr*****", cptValuesStr)
        # print("*****cptValuesList*****", cptValuesList)

        # 3. cpt Dict
        cptDict = {}
        parentNum = len(nodeInfo.getElementsByTagName("GIVEN"))
        cptKeysList = createTFKeysWithNum(parentNum)
        for i in range(len(cptKeysList)):
            if len(cptKeysList[i]) == 1:
                cptDict[cptKeysList[i][0]] = float(cptValuesList[i])
            elif len(cptKeysList[i]) > 1:
                cptDict[cptKeysList[i]] = float(cptValuesList[i])

        # print("*****cptDict*****", cptDict)
        if parentNum == 0:
            nodeTupleList.append(tuple([varName, parentsName, float(cptValuesList[0])]))
        else:
            nodeTupleList.append(tuple([varName, parentsName, cptDict]))

    # print("*****nodeTupleList*****", nodeTupleList)
        
    return BayesNet(nodeTupleList)




def createTFKeysWithNum(num):
    """
    returns a list containing True/False tuples

    """
    rtList = []

    for i in range(num):
        if len(rtList) == 0:
            rtList.append([T])
            rtList.append([F])
        else:
            rtListToRenew = []
            for item in rtList:
                itemT = item + [T]
                itemF = item + [F]
                rtListToRenew.append(itemT)
                rtListToRenew.append(itemF)
            rtList = rtListToRenew
    rtList = [tuple(iList) for iList in rtList]
    return rtList


# util functions ****************************************************************************
  
T, F = True, False












