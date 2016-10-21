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
import datetime

#Being used  by Reactive Explorer
percepts = {'stench' : 0, 'breeze' : 0, 'glitter' : 0, 'bump' : 0, 'scream' : 0, 'death' : 0}
goldCount = 0
killedByWumpus = 0
fellIntoPit = 0
killedWumpus = 0

'''
Explorer Class
'''
class Explorer:

    def __init__(self, world):
        # Associate world with this explorer
        self.world = world
        # Initial Score
        self.score = 0
        # Randomize initial orientation
        self.orientation = random.choice(['N','S','E','W'])
        # Count wumpuses and give the explorer that many arrows, also init location
        arrowCount = 0
        for i in range(len(world)):
            for j in range(len(world)):
                if world[i][j] == "W":
                    arrowCount += 1
                elif world[i][j] == "x":
                    self.x = i
                    self.y = j 
                
        self.arrowCount = arrowCount
        
        # Initial Knowledge Base as a blank list
        self.kb = []
        
        # init the list of percepts to 0 first
        self.list_percepts = {'Stench' : 0, 'Breeze' : 0, 'Glitter' : 0, 'Bump' : 0, 'Scream' : 0, 'Death' : 0}
        
    def add_rule(self, rule):
        ''' Add a rule (IN CLAUSE FORM) to the explorer's KB'''
        if rule not in self.kb:
            self.kb.append(rule)
        
    def update_kb(self):
        ''' Turns current percepts into clauses for KB'''
        #Position
        self.add_rule([[(2,"Visited"),(2,str((self.x,self.y)))]])
        #Stench
        if self.list_percepts['Stench']:
            self.add_rule([[(2,"Stench"),(2,str((self.x,self.y)))]])
            self.add_rule([[(2,"Wumpus"),(2,str(((self.x-1),self.y)))],[(2,"Wumpus"),(2,str(((self.x+1),self.y)))],[(2,"Wumpus"),(2,str((self.x,(self.y+1))))],[(2,"Wumpus"),(2,str((self.x,(self.y-1))))]])
        else:
            self.add_rule([["NOT",(2,"Stench"),(2,str((self.x,self.y)))]])
        #Breeze
        if self.list_percepts['Breeze']:
            self.add_rule([[(2,"Breeze"),(2,str((self.x,self.y)))]])
            self.add_rule([[(2,"Pit"),(2,str(((self.x-1),self.y)))],[(2,"Pit"),(2,str(((self.x+1),self.y)))],[(2,"Pit"),(2,str((self.x,(self.y+1))))],[(2,"Pit"),(2,str((self.x,(self.y-1))))]])
        else:
            self.add_rule([["NOT",(2,"Breeze"),(2,str((self.x,self.y)))]])
        #Glitter
        if self.list_percepts['Glitter']:
            self.add_rule([[(2,"Glitter"),(2,str((self.x,self.y)))]])
#         else:
#             self.add_rule([["NOT",(2,"Glitter"),(2,str((self.x,self.y)))]])
        #Bump
        if self.list_percepts['Bump']:
            if self.orientation == "N":
                self.add_rule([[(2,"Obstacle"),(2,str(((self.x-1),self.y)))]])
                self.add_rule([[(2,"Visited"),(2,str(((self.x-1),self.y)))]])
            elif self.orientation == "S":
                self.add_rule([[(2,"Obstacle"),(2,str(((self.x+1),self.y)))]])
                self.add_rule([[(2,"Visited"),(2,str(((self.x+1),self.y)))]])
            elif self.orientation == "E":
                self.add_rule([[(2,"Obstacle"),(2,str((self.x,(self.y+1))))]])
                self.add_rule([[(2,"Visited"),(2,str((self.x,(self.y+1))))]])
            elif self.orientation == "W":
                self.add_rule([[(2,"Obstacle"),(2,str((self.x,(self.y-1))))]])
                self.add_rule([[(2,"Visited"),(2,str((self.x,(self.y+1))))]])
#         else:
#             self.add_rule([["NOT",(2,"Bump"),(2,str(self.orientation)),(2,str((self.x,self.y)))]])
        

    def update_percepts(self) : 
        '''
        checking the values of adjacent cells so as to update the stench and breeze because for other percepts you need to be on 
        block to identify it
        '''
        adj_cells = adjCells(self.x,self.y,len(self.world))
        noWumpus = True
        noPit = True
        noGlitter = True
        noDeath = True
        noObstacle = True
        for values in adj_cells :
            if self.world[values[0]][values[1]] == 'W':
                self.list_percepts['Stench'] = 1
                noWumpus = False
            if self.world[values[0]][values[1]] == 'P':
                self.list_percepts['Breeze'] = 1
                noPit = False
        if noWumpus :
            self.list_percepts['Stench'] = 0
        if noPit :
            self.list_percepts['Breeze'] = 0
        '''
        this area will check the current cell value for wumpus, pit, glitter
        we are calling if else each time to make sure it doesn't stick to one in the percept dictionary
        '''
        if self.world[self.x][self.y] == 'G':
            self.list_percepts['Glitter'] = 1
        elif self.world[self.x][self.y] != 'G':
            self.list_percepts['Glitter'] = 0
        if self.world[self.x][self.y] == 'W' or self.world[self.x][self.y] == 'P':
            self.list_percepts['Death'] = 1
        if self.world[self.x][self.y] != 'W':
            self.list_percepts['Death'] = 0
        if self.world[self.x][self.y] != 'P':
            self.list_percepts['Death'] = 0
        if self.world[self.x][self.y] == 'O':
            self.list_percepts['Bump'] = 1
        elif self.world[self.x][self.y] != 'O':
            self.list_percepts['Bump'] = 0
        self.list_percepts['Scream'] = 0       
        
    def turn_left(self):
        if self.orientation == 'N' : 
            self.orientation = 'W'
        elif self.orientation == 'E' : 
            self.orientation = 'N'
        elif self.orientation == 'S' :
            self.orientation = 'E'
        else :
            self.orientation = 'S'
        self.score += (-1)
    
    def turn_right(self):
        if self.orientation == 'N' : 
            self.orientation = 'E'
        elif self.orientation == 'E' : 
            self.orientation = 'S'
        elif self.orientation == 'S' :
            self.orientation = 'W'
        else :
            self.orientation = 'N'
        self.score += (-1)
    
    def forward(self):
        print("* Moving " + str(self.orientation))
        row = self.x
        column = self.y
        if self.orientation == 'N' :
            if (row - 1) < len(self.world) and (row - 1) >= 0:
                self.x = row - 1
            else:
                percepts['bump'] = 1
                self.list_percepts['Bump'] = 1
                self.update_kb()
        elif self.orientation == 'E' : 
            if (column + 1) < len(self.world) and (column + 1) >= 0:
                self.y = column + 1
            else:
                percepts['bump'] = 1
                self.list_percepts['Bump'] = 1
                self.update_kb()
        elif self.orientation == 'S' :
            if (row + 1) < len(self.world) and (row + 1) >= 0:
                self.x = row + 1
            else:
                percepts['bump'] = 1
                self.list_percepts['Bump'] = 1
                self.update_kb()
        else :
            if (column - 1) < len(self.world) and (column - 1) >= 0:
                self.y = column - 1
            else:
                percepts['bump'] = 1
                self.list_percepts['Bump'] = 1
                self.update_kb()
        self.score += (-1)
        
        
        self.update_percepts()
        
        if self.world[self.x][self.y] == 'O' :
            print("Obstacle, go back to the previous cell")
            self.x = row
            self.y = column 
            self.list_percepts['Bump'] = 1
            self.update_kb()
            self.update_percepts()
            percepts['bump'] = 1
            self.score += (-1)
        elif self.world[self.x][self.y] == 'W' :
            print("Killed by Wumpus! -1000 score and go back to the previous cell")
            self.add_rule([[(2,"Wumpus"),(2,str((self.x,self.y)))]])
            self.x = row
            self.y = column
            self.score += (-1000)  
        elif self.world[self.x][self.y] == 'P' :
            print("Killed by Pit! -1000 score and go back to the previous cell")
            self.add_rule([[(2,"Pit"),(2,str((self.x,self.y)))]])
            self.x = row
            self.y = column
            self.score += (-1000)
        
    def shoot(self):
        print("* Fire!")
        self.score -= 10
        self.arrowCount -= 1
        location_x = self.x 
        location_y = self.y
        orientation = self.orientation
        if orientation == 'N' :
            for i in range(location_x) : 
                #When we kill a wumpus we display it as a cross '+' 
                if self.world[location_x-i][location_y] == 'O':
                    break
                elif self.world[location_x-i][location_y] == 'W' :
                    print("*HIT*")
                    self.world[location_x-i][location_y] = '+'
                    self.list_percepts['Scream'] = 1
                    self.score += 10
                    percepts['scream'] = 1  
        elif orientation == 'S' :
            for i in range(len(self.world)-location_x) : 
                #When we kill a wumpus we display it as a cross '+' 
                if self.world[location_x+i][location_y] == 'O':
                    break
                elif self.world[location_x+i][location_y] == 'W' :
                    print("*HIT*")
                    self.world[location_x+i][location_y] = '+'
                    self.list_percepts['Scream'] = 1
                    self.score += 10
                    percepts['scream'] = 1  
        elif orientation == 'E' : 
            for i in range(len(self.world)-location_y) : 
                #When we kill a wumpus we display it as a cross '+' 
                if self.world[location_x][location_y+i] == 'O':
                    break
                elif self.world[location_x][location_y+i] == 'W' :
                    print("*HIT*")
                    self.world[location_x][location_y+i] = '+'
                    self.list_percepts['Scream'] = 1
                    self.score += 10
                    percepts['scream'] = 1  
        else : 
            for i in range(location_y) : 
                #When we kill a wumpus we display it as a cross '+' 
                if self.world[location_x][location_y-i] == 'O':
                    break
                elif self.world[location_x][location_y-i] == 'W' :
                    print("*HIT*")
                    self.world[location_x][location_y-i] = '+'
                    self.list_percepts['Scream'] = 1
                    self.score += 10
                    percepts['scream'] = 1  
    
    def grab(self):
        if self.world[self.x][self.y] == 'G':
            self.list_percepts['Glitter'] = 1
            self.score += 1000
            print("GOLD FOUND! YOU WIN!")
        elif self.world[self.x][self.y] != 'G':
            self.list_percepts['Glitter'] = 0
 
'''
Reactive Explorer
Doesn't use any of the logic purely random based approaches to next cell.
Will know what is the current cell for instance the player will know whether there is stench/Breeze in cell 
But which cell to go is purely random. Player has no idea what is safe.
'''    
def reactiveExplorer(size):

    #Actions that the player can perform
    listofActions = ["grab","shoot","left","right","forward"]
    worldMatrix = worldCreation(size)
    printWorld(worldMatrix)
    #print(worldMatrix)
    reactiveCalls = Explorer(worldMatrix)
    setofUniqueCoordinates = set()
    totalMoves = 0
    while percepts['glitter'] != 1 or percepts['death'] != 1:
        print("**********************************************")
        #update percept scream to 0
        percepts['scream'] = 0
        percepts['bump'] = 0
        print("Your Score is " + str(reactiveCalls.score))
        print("You have " + str(reactiveCalls.arrowCount) + " arrow(s)")
        rowPosition = reactiveCalls.x
        columnPosition = reactiveCalls.y
        orientation = reactiveCalls.orientation
        setofUniqueCoordinates.add((rowPosition,columnPosition))
        #Printing values of row, column and orientation for testing purpose
        print("Your current row is " + str(rowPosition))
        print("Your current column is " + str(columnPosition))
        print("You are facing in " + str(orientation) + " Direction")
        print("Your percepts are " + str(percepts))
        gridSize = len(worldMatrix)
        currentState = adjCells(rowPosition,columnPosition,gridSize)
        '''
        print the adjacent matrix
        print(currentState)
        '''
        if worldMatrix[rowPosition][columnPosition] == 'G':
            percepts['glitter'] = 1
        elif worldMatrix[rowPosition][columnPosition] != 'G':
            percepts['glitter'] = 0  
        if worldMatrix[rowPosition][columnPosition] == 'W':
            percepts['death'] = 1
            print("You were killed by Wumpus")
            reactiveCalls.score = reactiveCalls.score - 1000
            print("Your final score after getting killed by Wumpus is" + str(reactiveCalls.score))
            global killedByWumpus
            killedByWumpus += 1
            break
        elif worldMatrix[rowPosition][columnPosition] != 'W':
            percepts['death'] = 0
            
        if worldMatrix[rowPosition][columnPosition] == 'P':
            percepts['death'] = 1
            print("You went into the pit")
            reactiveCalls.score = reactiveCalls.score - 1000
            print("Your final score after falling in pit is" + str(reactiveCalls.score))  
            global fellIntoPit
            fellIntoPit += 1       
            break
        elif worldMatrix[rowPosition][columnPosition] != 'P':
            percepts['death'] = 0
        '''
        To check the adjacent cells and find if there is stench or breeze in current cell
        '''
        for values in currentState:
            #print(worldMatrix[values[0]][values[1]])
            if worldMatrix[values[0]][values[1]] == 'W':
                percepts['stench'] = 1
            if worldMatrix[values[0]][values[1]] == 'P':
                percepts['breeze'] = 1
        
        print("Your percepts after checking adjacent value " + str(percepts))
        if percepts['glitter'] == 1:
            perceptAction = 'grab'
        elif (percepts['stench'] == 1 or percepts['breeze'] == 1):
            perceptAction = random.choice(listofActions[1:])
        else:
            perceptAction = random.choice(listofActions[2:])
        totalMoves += 1
        print("Your action is " + str(perceptAction))
        
        if perceptAction == 'left':
            reactiveCalls.turn_left()
            
        elif perceptAction == 'right':
            reactiveCalls.turn_right()
            
        elif perceptAction == 'forward':
            percepts['stench'] = 0
            percepts['breeze'] = 0
            reactiveCalls.forward()
            
        elif perceptAction == 'grab':
            print("You got the gold")
            reactiveCalls.score = reactiveCalls.score + 1000
            print("Your final score after getting gold is" + str(reactiveCalls.score))
            global goldCount
            goldCount += 1   
            break
        elif perceptAction == 'shoot' and reactiveCalls.arrowCount > 0:
            reactiveCalls.shoot()
            if percepts['scream'] == 1:
                global killedWumpus
                killedWumpus += 1  

            
        print("Your percepts after performing an action " + str(percepts))
            #Just in case we might not use the shoot in class we can use this
        '''
            if reactiveCalls.orientation == 'N' :
                for i in range(len(worldMatrix)) : 
                    if worldMatrix[i][columnPosition] == 'W' :
                        worldMatrix[i][columnPosition] = '+'
                        reactiveCalls.score += 10
                        percepts['scream'] = 1  
            elif orientation == 'E' :
                for i in range(len(worldMatrix)) : 
                    if worldMatrix[rowPosition][i] == 'W' :
                        worldMatrix[rowPosition][i] = '+'
                        reactiveCalls.score += 10
                        percepts['scream'] = 1       
            elif orientation == 'S' : 
                for i in range(len(worldMatrix)) : 
                    if worldMatrix[i][columnPosition] == 'W' :
                        worldMatrix[i][columnPosition] = '+'
                        reactiveCalls.score += 10
                        percepts['scream'] = 1
            else : 
                for i in range(len(worldMatrix)) : 
                    if worldMatrix[rowPosition][i] == 'W' :
                        worldMatrix[rowPosition][i] = '+'
                        reactiveCalls.score += 10
                        percepts['scream'] = 1
            reactiveCalls.arrowCount = reactiveCalls.arrowCount - 1
            if percepts['scream'] != 1:
                reactiveCalls.score += -10
        '''

        '''
        this area will check the current cell value for wumpus, pit, glitter
        we are calling if else each time to make sure it doesn't stick to 1 in the percept dictionary
        ''' 
    printWorld(worldMatrix)
    print("Set of unique coordinates traversed are " + str(setofUniqueCoordinates))
    print("Number of cell explored are " + str(len(setofUniqueCoordinates)))
    print("Total moves taken " + str(totalMoves))
    
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
#             if worldMaker[i][j] == 'P' : 
#                 adj_cell = adjCells(i,j,gridSize)
#                 for x in adj_cell :
#                     worldMaker[x[0]][x[1]] = 'B'
#             if worldMaker[i][j] == 'W' : 
#                 adj_cell = adjCells(i,j,gridSize)
#                 for x in adj_cell :
#                     worldMaker[x[0]][x[1]] = 'S'
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
Unification
'''            
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
        if not x or not y:
            return theta
        return try_unify(x[1:], y[1:], try_unify(x[0], y[0], theta))
    # Otherwise we fail
    return False
    
def unify_var(var,x,theta):
    if var in theta:
        return try_unify(theta[var], x, theta)
    elif type(x) is tuple and x in theta:
        return try_unify(var, theta[x], theta)
    # OCCURS CHECK MAYBE
    else:
        theta[var] = x
        return theta
    
def substitute(clause, theta):
    ''' Called after unification to apply the substitution strings to a
    rule
    '''
    new_clause = []
    for term in clause:
        new_term = []
        for var in term:
            new_var = var
            if var in theta:
                new_var = theta[var]
            new_term.append(new_var)
        new_clause.append(new_term)
    return new_clause

def fol_resolution(KB, alpha):
    '''Resolution algorithm performs FOL Resolution on all clauses in KB, the knowledge base,
    which is a list of sentences in clause form, and alpha, which is a new clause.
    Returns true if two clauses can resolve to the empty set and false otherwise.
    KB form: KB is a list [] of clauses, each clause connected by implicit "AND"
    Clause form: Each clause is a list [] of terms, each term connected by implicit "OR"
    Term form: Each term is a list [] of tuples (), first value of tuple is integer code and 2nd value is string representation
    '''
    #print("Alpha: " + str(alpha))
    cl = KB.copy()
    #negate(alpha)
    cl.append(alpha)
    #print(cl)
    # Sort to attempt unit resolution
    clauses = sorted(cl, key=len)
    #print(clauses)
    while True:  
        new = []
        for i in range(len(clauses)):
            for j in range(i, len(clauses)):
                if i != j:
                    resolvants = fol_resolve(clauses[i],clauses[j])
                    # If return empty clause then return true
                    if resolvants:
                        if not resolvants[0]:
                            return True
                        # Union resolvants with new
                        union(new, resolvants)
        # Subset check - if no new resolvants have been created, resolution fails
        if subsetCheck(new, clauses):
            return False
        # Union new resolvants into clauses
        union(clauses, new)
        # Prune clauses
        #prune(clauses)
                               
def fol_resolve(C1, C2):
    #print("Resolve " + str(C1) + " and " + str(C2))
    mgu = False
    for term in C1:
        for term2 in C2:
            # Try to unify two terms iff one is negated
            if term[0] == "NOT" and term2[0] != "NOT":
                # Remove negation for unification algorithm, reapply after
                negate(term)
                mgu = try_unify(term, term2, {})
                negate(term)
            elif term[0] != "NOT" and term2[0] == "NOT":
                # Remove negation for unification algorithm, reapply after
                negate(term2)
                mgu = try_unify(term, term2, {})
                negate(term2)
            # If they unify, remove them from resolvant and apply mgu substitution
            if mgu != False:
                c1copy = C1.copy()
                c1copy.remove(term)
                c2copy = C2.copy()
                c2copy.remove(term2)
                c1copy.extend(c2copy)
                sub = substitute(c1copy, mgu)
                #if not sub:
                #    print("C1: " + str(C1))
                #    print("C2: " + str(C2))
                return [sub]
    # No resolvants possible for two clauses
    return False

def negate(alpha):
    '''Takes a term, alpha, in the form of a list of tuples, and adds "NOT" as first item in list, or removes it if
    it is already there
    '''
    if alpha[0] == "NOT":
        alpha.pop(0)
    else:
        alpha.insert(0,"NOT")
        
def union(S1, S2):
    '''Perform union function on two lists, adding unique items from S2 into S1
    '''
    for item in S2:
        if item not in S1:
            S1.append(item)
            
def subsetCheck(S1, S2):
    ''' Determine if S1 is a subset of S2
    '''
    isSubset = True
    for i in range(len(S1)):
        if S1[i] not in S2:
            isSubset = False
    return isSubset
            
def prune(KB):
    '''Helper function for resolution, prunes clauses from a KB that have two complementary literals,
    also removes copies to try and ensure KB remains a set
    '''
    removalIndex = set()
    newKB = []
    
    for i in range(len(KB)):
        for j in range(i, len(KB)):
            if i != j:
                # Remove copies
                if KB[i] == KB[j]:
                    removalIndex.add(j)
    for i in range(len(KB)):
        if i not in removalIndex:
            newKB.append(KB[i])
            
def buildRule():
    print("***BUILDING A CLAUSE***")
    rule = []
    t = input("Enter number of terms: ")
    numTerms = int(t)
    for i in range(numTerms):
        term = []
        print("-Term " + str(i+1))
        neg = input("Negated? (Y/N): ")
        if neg.lower() == "y":
            term.append("NOT")
        pred = input("Enter predicate: ")
        term.append((2,pred))
        a = input("Enter number of arguments for predicate: ")
        numArgs = int(a)
        for j in range(numArgs):
            print("--Arg " + str(j+1))
            isvar = input("Is it a variable or a constant? (V/C): ")
            arg = input("Enter name of argument: ")
            if isvar.lower() == "v":
                term.append((1,arg))
            elif isvar.lower() == "c":
                term.append((2,arg))
        rule.append(term)
    print(rule)
    return rule



def logicExplorer():
    # Test small world with no pits, wumpus, or obstacles
    world = worldCreation()
    printWorld(world)
    exp = Explorer(world)
    
    ''' DEFINE EDGES OF WORLD IN KB '''
    for j in range(len(world)):
        # Mark outside of world as obstacle and visited to work with exploration
        exp.add_rule([[(2,"Obstacle"),(2,str((-1,j)))]])
        exp.add_rule([[(2,"Visited"),(2,str((-1,j)))]])
    for j in range(len(world)):
        # Mark outside of world as obstacle and visited to work with exploration
        exp.add_rule([[(2,"Obstacle"),(2,str((len(world),j)))]])
        exp.add_rule([[(2,"Visited"),(2,str((len(world),j)))]])
    for j in range(len(world)):
        # Mark outside of world as obstacle and visited to work with exploration
        exp.add_rule([[(2,"Obstacle"),(2,str((j,-1)))]])
        exp.add_rule([[(2,"Visited"),(2,str((j,-1)))]])
    for j in range(len(world)):
        # Mark outside of world as obstacle and visited to work with exploration
        exp.add_rule([[(2,"Obstacle"),(2,str((j,len(world))))]])
        exp.add_rule([[(2,"Visited"),(2,str((j,len(world))))]])
    
    ''' KNOWLEDGE BASE'''
    exp.add_rule([["NOT",(2,"Glitter"),(1,"x1")],[(2,"Action"),(2,"Grab")]])
    
    exp.add_rule([[(2,"Stench"),(1,"x2")],[(2,"Breeze"),(1,"x2")],[(2,"Safe"),(2,"N"),(1,"x2")]])
#     exp.add_rule([["NOT",(2,"Safe"),(2,"N"),(1,"x21")],["NOT",(2,"Breeze"),(1,"x21")]])
#     exp.add_rule([["NOT",(2,"Safe"),(2,"N"),(1,"x22")],["NOT",(2,"Stench"),(1,"x22")]])
    
    exp.add_rule([[(2,"Stench"),(1,"x3")],[(2,"Breeze"),(1,"x3")],[(2,"Safe"),(2,"S"),(1,"x3")]])
#     exp.add_rule([["NOT",(2,"Safe"),(2,"S"),(1,"x31")],["NOT",(2,"Breeze"),(1,"x31")]])
#     exp.add_rule([["NOT",(2,"Safe"),(2,"S"),(1,"x32")],["NOT",(2,"Stench"),(1,"x32")]])
    
    exp.add_rule([[(2,"Stench"),(1,"x4")],[(2,"Breeze"),(1,"x4")],[(2,"Safe"),(2,"E"),(1,"x4")]])
#     exp.add_rule([["NOT",(2,"Safe"),(2,"E"),(1,"x42")],["NOT",(2,"Breeze"),(1,"x42")]])
#     exp.add_rule([["NOT",(2,"Safe"),(2,"E"),(1,"x43")],["NOT",(2,"Stench"),(1,"x43")]])
    
    exp.add_rule([[(2,"Stench"),(1,"x5")],[(2,"Breeze"),(1,"x5")],[(2,"Safe"),(2,"W"),(1,"x5")]])
#     exp.add_rule([["NOT",(2,"Safe"),(2,"W"),(1,"x52")],["NOT",(2,"Breeze"),(1,"x52")]])
#     exp.add_rule([["NOT",(2,"Safe"),(2,"W"),(1,"x53")],["NOT",(2,"Stench"),(1,"x53")]])

    exp.add_rule([["NOT",(2,"Safe"),(1,"x6")],["NOT",(2,"Wumpus"),(1,"x6")]])
    exp.add_rule([["NOT",(2,"Safe"),(1,"x7")],["NOT",(2,"Pit"),(1,"x7")]])
    exp.add_rule([["NOT",(2,"Obstacle"),(1,"x8")],["NOT",(2,"Wumpus"),(1,"x8")]])
    exp.add_rule([["NOT",(2,"Obstacle"),(1,"x9")],["NOT",(2,"Pit"),(1,"x9")]])
    
    #exp.add_rule([["NOT",(2,"Stench"),(1,"x6")],[(2,"Wumpus"),(2,"N"),(1,"x6")],[(2,"Wumpus"),(2,"S"),(1,"x6")],[(2,"Wumpus"),(2,"E"),(1,"x6")],[(2,"Wumpus"),(2,"W"),(1,"x6")]])
    
    ''' GAME LOOP'''
    gameover = False
    while not gameover:
        # Print Info
        print("______________________________")
        print("X,Y: " + str(exp.x) + str(exp.y))
        print("Orientation: " + str(exp.orientation))
        print("Score: " + str(exp.score))
        
        # Update Percepts
        print("Updating percepts...")
        exp.update_percepts()
        exp.update_kb()
        
        # Define Surroundings
        nextCell = None
        rightCell = None
        leftCell = None
        
        if exp.orientation == "N":
            nextCell = (exp.x-1,exp.y)
            rightCell = (exp.x,exp.y+1)
            leftCell = (exp.x,exp.y-1)
        elif exp.orientation == "S":
            nextCell = (exp.x+1,exp.y)
            rightCell = (exp.x,exp.y-1)
            leftCell = (exp.x,exp.y+1)
        elif exp.orientation == "E":
            nextCell = (exp.x,exp.y+1)
            rightCell = (exp.x+1,exp.y)
            leftCell = (exp.x-1,exp.y)
        elif exp.orientation == "W":
            nextCell = (exp.x,exp.y-1)
            rightCell = (exp.x-1,exp.y)
            leftCell = (exp.x+1,exp.y)
            
        #Mark Safe Squares
        alpha = [["NOT",(2,"Safe"),(2,"N"),(2,str((exp.x,exp.y)))]]
        if fol_resolution(exp.kb, alpha):
            # North of current location safe
            exp.add_rule([[(2,"Safe"),(2,str(((exp.x-1),exp.y)))]])
        alpha = [["NOT",(2,"Safe"),(2,"S"),(2,str((exp.x,exp.y)))]]
        if fol_resolution(exp.kb, alpha):
            # South of current location safe
            exp.add_rule([[(2,"Safe"),(2,str(((exp.x+1),exp.y)))]])
        alpha = [["NOT",(2,"Safe"),(2,"E"),(2,str((exp.x,exp.y)))]]
        if fol_resolution(exp.kb, alpha):
            # East of current location safe
            exp.add_rule([[(2,"Safe"),(2,str((exp.x,(exp.y+1))))]])
        alpha = [["NOT",(2,"Safe"),(2,"W"),(2,str((exp.x,exp.y)))]]
        if fol_resolution(exp.kb, alpha):
            # West of current location safe
            exp.add_rule([[(2,"Safe"),(2,str((exp.x,(exp.y-1))))]])
        
        # SEARCH KB FOR BEST ACTION
        
        # Get the gold!
        alpha = [["NOT",(2,"Action"),(2,"Grab")]]
        if fol_resolution(exp.kb, alpha):
            # Best Action is Grab
            exp.grab()
            gameover = True
            continue
        
        # Explore!
        alpha = [["NOT",(2,"Obstacle"),(2,str(nextCell))]]
        if fol_resolution(exp.kb, alpha):
            # Impossible to move forward
            exp.turn_right()
        else:
            # Possible to move forward
            alpha = [["NOT",(2,"Safe"),(2,exp.orientation),(2,str((exp.x,exp.y)))]]
            if fol_resolution(exp.kb, alpha):
                # Safe to Move Forward
                alpha = [["NOT",(2,"Visited"),(2,str(nextCell))]]
                if fol_resolution(exp.kb, alpha):
                    # Already visited next cell, check other directions
                    alpha = [["NOT",(2,"Visited"),(2,str(rightCell))]]
                    if fol_resolution(exp.kb, alpha):
                        # Already visited cell on right, check left
                        alpha = [["NOT",(2,"Visited"),(2,str(leftCell))]]
                        if fol_resolution(exp.kb, alpha):
                            # Already visited all nearby cells so just move forward
                            exp.forward()
                        else:
                            # Left cell unvisited, let's go there
                            exp.turn_left()
                            exp.forward()
                    else:
                        # Right cell unvisited, let's go there
                        exp.turn_right()
                        exp.forward()                 
                else:
                    # Next cell unvisited, let's go there
                    exp.forward()
            else:
                # Not safe to move forward
                print("Not safe!!")
                alpha = [["NOT",(2,"Pit"),(2,str(nextCell))]]
                if fol_resolution(exp.kb, alpha):
                    # Pit ahead, turn away!
                    exp.turn_right()
                else:
                    alpha = [["NOT",(2,"Wumpus"),(2,str(nextCell))]]
                    if fol_resolution(exp.kb, alpha):
                        # Wumpus ahead, take him out!
                        if exp.arrowCount > 0:
                            exp.shoot()
                            exp.forward()
                        else:
                            # Run away!
                            exp.turn_right()
                    else:
                        # No pit or wumpus detected
                        exp.forward()
    
    print("***GAME OVER***")
    print("Final Score: " + str(exp.score))    
    
 

def resolutionTest():
    '''TEST 1'''
    # Note a KB will be a triple list!
    a = [[[(2, "Knows"),(2,"John")]]]
    # Note a clause will be a double list!
    b = [["NOT",(2, "Knows"),(2,"John")]]
    print("SHOULD BE TRUE - " + str(fol_resolution(a, b)))
    
    '''TEST 2'''
    # Var - 1 Constant - 2
    a = [
          [["NOT",(2,"American"),(1,"x1")],["NOT",(2,"Weapon"),(1,"y1")],["NOT",(2,"Sells"),(1,"x1"),(1,"y1"),(1,"z1")],["NOT",(2,"Hostile"),(1,"z1")],[(2,"Criminal"),(1,"x1")]],
          [["NOT",(2,"Missile"),(1,"x2")],["NOT",(2,"Owns"),(2,"Nono"),(1,"x2")],[(2,"Sells"),(2,"West"),(1,"x2"),(2,"Nono")]],
          [["NOT",(2,"Enemy"),(1,"x3"),(2,"America")],[(2,"Hostile"),(1,"x3")]],
          [["NOT",(2,"Missile"),(1,"x4")],[(2,"Weapon"),(1,"x4")]],
          [[(2,"Owns"),(2,"Nono"),(2,"M1")]],
          [[(2,"American"),(2,"West")]],
          [[(2,"Missile"),(2,"M1")]],
          [[(2,"Enemy"),(2,"Nono"),(2,"America")]]
        ]
    b = [["NOT",(2,"Criminal"),(2,"West")]]
    print("SHOULD BE TRUE - " + str(fol_resolution(a, b)))        
    
    '''TEST 3'''
    a = [
          [["NOT",(2,"Glitter"),(1,"x1")],["NOT",(2,"Player"),(1,"x1")],[(2,"Action"),(2,"Grab")]],
          [[(2,"Glitter"),(2,"(1,1)")]],
          [[(2,"Player"), (2,"(1,1)")]]
        ]
    b = [["NOT",(2,"Action"),(2,"Grab")]]
    print("SHOULD BE TRUE - " + str(fol_resolution(a, b)))  
    
    ''' TEST 4 USING RULE BUILDER'''
#     kb = []
#     for i in range(3):
#         newRule = buildRule()
#         kb.append(newRule)
#     
#     myRule = buildRule()
#     print("ATTEMPTING TO RESOLVE...SUCCESS? - " + str(fol_resolution(kb, myRule)))  
    
def worldGeneratorTest():
    # Generate 5x5 world with 10% chance of Pits, Obstacles, and Wumpi
    worldMatrix = worldCreation()
    printWorld(worldMatrix)
    myExplorer = Explorer(worldMatrix)
    print("Explorer Arrows: " + str(myExplorer.arrowCount))
    print("Explorer Location: " + str(myExplorer.x) + ", " + str(myExplorer.y))
    print("Explorer Orientation: " + str(myExplorer.orientation))
    
    # Generate 25x25 world with 5% chance of pits, obs, and wumpi
    #newWorld = worldCreation(25, .05, .05, .05)
    #printWorld(newWorld)
    
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
    x = [(2, "Knows"),(2,"John"),(1,"x")]
    y = [(2, "Knows"),(1,"x"),(2,"Elizabeth")]
    print(try_unify(x, y, {}))

    # Testing unification + substitution
    x = [(2, "Knows"),(2,"John"),(1,"x")]
    y = [(2, "Knows"),(1,"y"),(2,"Bill")]
    theta = try_unify(x, y, {})
    print(x)
    print(theta)
    print(substitute([x], theta))

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

#rule2 = "W" + coordinates + "o" 
#rules = ["", ""]

def get_time():
    return datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')

def testRuns():
    print("Test runs to demonstrate Reactive Explorer running 10 times on different sizes")
    for matrixSize in range(5, 30, 5):
        global goldCount
        goldCount = 0
        global killedByWumpus
        killedByWumpus = 0
        global fellIntoPit
        fellIntoPit = 0
        global killedWumpus
        killedWumpus = 0
        for runNum in range(0, 10):
            reactiveExplorerTest(matrixSize,runNum)

def reactiveExplorerTest(matrixSize, runNum):
    print("############################################")
    print("Start time: " + str(get_time()))
    print("--------------------------------------------")
    print("Test case number " + str(runNum))
    print("Grid size is " + str(matrixSize) + " x " + str(matrixSize))
    reactiveExplorer(matrixSize)
    print("Total number of times gold taken " + str(goldCount))
    print("Total number of times killed by wumpus " + str(killedByWumpus))
    print("Total number of time fell into the pit " + str(fellIntoPit))
    print("Total number of time killed the wumpus " + str(killedWumpus))
    print("--------------------------------------------")
    print("End time: " + str(get_time()))
        
def main():
    #worldGeneratorTest()
    #unificationTest()
    #resolutionTest()
    logicExplorer()
    #reactiveExplorer(5)
    #testRuns()
    
    
if __name__ == '__main__':
    main()
