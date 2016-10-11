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

def adjCells(x,y,gridSize):
	list_adj = list()
	if (x+1) < gridSize : 
		list_adj.append([x+1,y])
	if (x-1) >= 0 : 
		list_adj.append([x-1,y])
	if (y+1) < gridSize : 
	    list_adj.append([x,y+1])
	if (y-1) >= 0 :
	    list_adj.append([x,y-1])
	return list_adj

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
            if worldMaker[i][j] == 'P' : 
                adj_cell = adjCells(i,j,gridSize)
                for x in adj_cell :
                    worldMaker[x[0]][x[1]] = 'B'
            if worldMaker[i][j] == 'W' : 
                adj_cell = adjCells(i,j,gridSize)
                for x in adj_cell :
                    worldMaker[x[0]][x[1]] = 'S'
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
    #print(type(var))
    #print(type(x))
    #print(type(theta))
    #if type(var) == type(x):
    if var in theta:
        return try_unify(theta[var], x, theta)
    elif type(x) is tuple and x in theta:
        return try_unify(var, theta[x], theta)
    # OCCURS CHECK MAYBE
    else:
        theta[var] = x
        return theta
#     else:
#         if type(x[0]) is tuple:
#             if x[0][0] == 1:
#                 x = (x[0][1],x[1][1])
#         if type(x[1]) is tuple:
#             if x[1][0] == 1:
#                 x = (x[1][1],x[0][1])
#         return unify_var(var, x, theta)
            
def try_unify(x,y,theta):
    ''' Attempt at unification algorithm.  Tuples for atomic vars, (int, string)
    as all possible inputs - var (1), constant/predicate(2)
    List of tuples for compound/strings
    return theta as dictionary of substitutions {var:val}
    '''
    if theta == False: return False
    if type(x) is tuple and type(y) is tuple:
        # String equality for atomic terms
        if x[1] == y[1]: return theta
    # X or Y is variable (code 1)
    if type(x) is tuple:
        if x[0] == 1:
            return unify_var(x, y, theta)
    if type(y) is tuple:
        if y[0] == 1:
            return unify_var(y, x, theta)
    # X and Y are compound/lists (do these need to be handled differently???)
    elif type(x) is list and type(y) is list:
        # Empty list check
        if not x and not y:
            return theta
        return try_unify(x[1:], y[1:], try_unify(x[0], y[0], theta))
    # Otherwise we fail
    else: return False
    
def worldGeneratorTest():
    # Generate 5x5 world with 10% chance of Pits, Obstacles, and Wumpi
    worldMatrix = worldCreation()
    printWorld(worldMatrix)
    
    # Generate 25x25 world with 5% chance of pits, obs, and wumpi
    newWorld = worldCreation(25, .05, .05, .05)
    printWorld(newWorld)
    
def unificationTest():
    # Example unification 
    # - all terms are tuples with integer code 1 for var, 2 for constant,
    # - place compound terms in a list
    x = (1, "x")
    y = (2, "joe")
    print(try_unify(x, y, {}))
    
    # Examples from book page 332
    x = [(2, "Knows"),(2,"John"),(1,"x")]
    y = [(2, "Knows"),(2,"John"),(2,"Jane")]
    print(try_unify(x, y, {}))
    
    x = [(2, "Knows"),(2,"John"),(1,"x")]
    y = [(2, "Knows"),(1,"y"),(2,"Bill")]
    print(try_unify(x, y, {}))
    
    # Working, doesn't unify interior terms but we can just substitute later if needed
    x = [(2, "Knows"),(2,"John"),(1,"x")]
    y = [(2, "Knows"),(1,"y"),[(2,"Mother"),(1,"y")]]
    print(try_unify(x, y, {}))
    
    # Should fail, untested
#     x = [(2, "Knows"),(2,"John"),(1,"x")]
#     y = [(2, "Knows"),(1,"x"),(2,"Elizabeth")]
#     print(try_unify(x, y, {}))

coordinates = [2]
#W = Wumpus
#o = or
#a = and
#I = Imply
#B = Breeze
#Gl = Glitter
#Go = Gold
#O = obstacle 
#P = Pit 
#Bu = Bump
#A = arrow
#K = Kill
#D = Dead 
#V = Victory
#S = Stench 
#C = Current position
#G = Grab
#L = Left
#R = Right
#M = Move
#Sh = Shoot
#Sc = Scream 

rule2 = "W" + coordinates + "o" 
rules = ["", ""]

        
def main():
    worldGeneratorTest()
    #unificationTest()
    
    
if __name__ == '__main__':
    main()
