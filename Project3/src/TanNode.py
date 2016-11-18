'''
CSCI 446 
Fall 2016
Project 3
Group 3

@author: Joe DeBruycker
@author: Shriyansh Kothari
@author: Sara Ounissi
'''
class TanNode(object):
    '''
    A custom node for a Tree data structure required for Project 3.
    Contains data structures and methods for tracking child nodes and 
    tree traversal during the classification task for TAN.
    '''
    def __init__(self, feature):
        ''' Constructor builds an empty node representing feature index'''
        self.children = list()
        self.childNames = list()
        self.weights = {}
        self.name = feature
        
    def addUndirectedEdge(self, neighborNode):
        ''' Makes an undirected edge between this node and neighbor by adding each
        to the other's child list, add names to namelist'''
        self.children.append(neighborNode)
        self.childNames.append(neighborNode.name)
        neighborNode.children.append(self)
        neighborNode.childNames.append(self.name)
        
    def setWeight(self, neighborIndex, weight):
        ''' Sets the weight for the edge between this node and the neighbor'''
        self.weights[neighborIndex] = weight
        
        