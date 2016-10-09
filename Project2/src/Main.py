'''
CSCI 446 
Fall 2016
Project 2
Group 3

@author: Joe DeBruycker
@author: Shriyansh Kothari
@author: Sara Ounissi
'''

import random
import sys
from _ast import Expr

def worldCreation(gridSize = 5, pPit = 0.1, pObstacle = 0.1, pWumpus = 0.1):
    '''  Generates a grid of rooms that make up a Wumpus World Cave instance.
    Each room is either Empty, or contains: a pit, a wumpus, an obstacle, or
    the gold (the goal state).  Inputs are gridSize, where size n creates an [nXn]
    grid, pPit:  probability of generating a pit, pObstacle:  probability of 
    generating an obstacle, and pWumpus, the probability of generating a 
    Wumpus.  Returns a reference to the grid.
    '''
    worldMaker = []
    emptyCount = 0
    
    #Print statements for Experimental and further design documents too
    #print("Currently selected size of Wumpus World is " + str(gridSize) + " x " + str(gridSize))
    
    for i in range(gridSize):
        worldMaker.append([])
        for j in range(gridSize):
            # Procedural World Generation
            if random.random() < pPit:
                worldMaker[i].append("P")
            elif random.random() < pObstacle:
                worldMaker[i].append("O")
            elif random.random() < pWumpus:
                worldMaker[i].append("W")
            else: 
                worldMaker[i].append("-")
                emptyCount += 1
                
    # Pick random start position and gold location from empty cells
    startCell = random.randint(1, emptyCount)
    for i in range(gridSize):
        for j in range(gridSize):
            if worldMaker[i][j] == '-':
                startCell -= 1
                if startCell == 0:
                    worldMaker[i][j] = 'x'
                    
    goldCell = random.randint(1, emptyCount-1)
    for i in range(gridSize):
        for j in range(gridSize):
            if worldMaker[i][j] == '-':
                goldCell -= 1
                if goldCell == 0:
                    worldMaker[i][j] = 'G'
            
    return worldMaker


def printWorld(world):
    ''' Prints a representation of the current Wumpus World.
    Key:    - - Empty Space
            x - Player (maybe use > < ^ V to represent direction?)
            O - Obstacle
            W - Wumpus
            P - Pit
            G - Gold
    '''
    print("Current Wumpus World: (" + str(len(world)) + "x" + str(len(world)) + ")")
    for i in range(len(world)):
        print('|', end=' ')
        for j in range(len(world)):
            print(world[i][j], end=' ')
        print('|')
        #print(sys.version)

'''
Unification Algorithm
'''
def unify(x,y,theta):
    #not using type() because type fails whereas isinstance tells properly about type https://stereochro.me/ideas/type-vs-isinstance
    
    if theta is None:
        return None
    elif x == y:
        return theta
    elif Variable(x):
        return unify_var(x,y,theta)
    elif Variable(y):
        return unify_var(y,x,theta)
    #Compound
    elif isinstance(x, Expr) and isinstance(y, Expr):
        '''
        doSomething for compound variable
        http://www.ibm.com/support/knowledgecenter/SSLTBW_2.1.0/com.ibm.zos.v2r1.ikjc300/ikj2g2_What_is_a_Compound_Variable_.htm
        '''
        return 
    #List
    elif type(x) is list and type(y) is list:
        return unify(x[1:], y[1:], unify(x[0], y[0], theta))
        #doSomething
    else:
        return None               
                
                    
def unify_var(var,x,theta):
        
def main():
    # Generate 5x5 world with 10% chance of Pits, Obstacles, and Wumpi
    worldMatrix = worldCreation()
    
    printWorld(worldMatrix)
    
    # Generate 25x25 world with 5% chance of pits, obs, and wumpi
    newWorld = worldCreation(25, .05, .05, .05)
    printWorld(newWorld)
    
if __name__ == '__main__':
    main()
