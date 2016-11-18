'''
CSCI 446 
Fall 2016
Project 3
Group 3

@author: Joe DeBruycker
@author: Shriyansh Kothari
@author: Sara Ounissi
'''
class ID3Node(object):
    '''
    A custom node for a Tree data structure required for Project 3.
    Contains data structures and methods for tracking child nodes and 
    tree traversal during the classification task for ID-3.
    '''
    def __init__(self):
        ''' Constructor'''
        self.children = list()
        self.featureValues = list()
        self.feature = None
        self.label = None
        self.prune_label = None
        
    def test(self, example):
        ''' Takes an example data instance and uses the decision tree node to return a class label
        or the child node that testing on the current node's feature indicates'''
        if self.label:
            return self.label
        else:
            for i in range(len(self.featureValues)):
                if example[self.feature] == self.featureValues[i]:
                    return self.children[i].test(example)
                
    def addChild(self, childNode, childValue):
        ''' Adds a new child node to this tree with a corresponding value'''
        self.children.append(childNode)
        self.featureValues.append(childValue)
    
    def setLabel(self, newLabel):
        ''' Sets class label which test will return'''
        self.label = newLabel
        
    def setFeature(self, newFeature):
        ''' Sets index on which test will operate '''
        self.feature = newFeature
        
    def setPruneLabel(self,newLabel):
        self.prune_label = newLabel
        
        