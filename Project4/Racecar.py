'''
CSCI 446 
Fall 2016
Project 4
Group 3

@author: Joe DeBruycker
@author: Shriyansh Kothari
@author: Sara Ounissi
'''

class Racecar():
    ''' Defines a class representing a racecar agent which navigates the environment
    defined by a Track object.  Keeps track of current position as coordinates in the 
    track as well as the control variables for acceleration and the derived velocity
    along both axes.  
    '''
    
    def __init__(self, startX, startY):
        ''' Initializes a Racecar object at a starting position on a Track '''
        self.x = startX 
        self.y = startY 
        # Initialize acceleration values as 0
        self.aX = 0
        self.aY = 0 
        # Initialize velocity values as 0
        self.vX = 0
        self.vY = 0