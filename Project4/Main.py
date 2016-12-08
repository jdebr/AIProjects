'''
CSCI 446 
Fall 2016
Project 4
Group 3

@author: Joe DeBruycker
@author: Shriyansh Kothari
@author: Sara Ounissi
'''
from Track import Track
from Racecar import Racecar
'''
Things to keep in mind

accelerationList = [-1,0,1]
punishment = -1
racer_State = {}
racer_Reward = {}
'''
'''
80% we accelerate hence 0.8 
20% no acceleration hence 0.2
'''
possibleActions = [(1,0),(1,1),(0,1),(-1,0),(-1,-1),(0,-1),(1,-1),(-1,1)]
raceTrack = []
states = defaultdict(list)
startStates = []
punishment = -1
discount = 0.4
stateValue = {}
reward = {}

def valueIteration(epsilon = 0.01):

    statesCopy = dict([(s,0) for s in racerStates()])
    discount = 0.9
    while True:
        sCopy = statesCopy.copy()
        delta = 0
        for s in racerStates():
            statesCopy[s] = giveRewards(s) + discount * max([sum([p * sCopy[s1] for(p, s1) in stateTransitions(s,a)]) for a in actions(s)])
            delta = max(delta, abs(statesCopy[s] - sCopy[s]))
            if delta < epsilon * (1-discount) / discount:
                return sCopy

def policyIdentification(sCopy):
    pi = {}
    for s in racerStates():
        pi[s] = argmax(actions(s), lambda a:utilityFunction(a,s,sCopy))
        
def utilityFunction(a,s,sCopy):
    return sum([p * sCopy[s1] for (p, s1) in stateTransitions(s, a)])

'''
Identify all possible states and appending to each state all the possible velocity it can have from -5 to +5 in both x and y
'''    
def racerStates():
    for row in range(len(raceTrack)):
        for column in range(len(raceTrack)):
            if(raceTrack[row][column] != '#'):
                state = (row,column)
                for xVelocity in range(-5,6):
                    for yVelocity in range(-5,6):
                        states[state].append((xVelocity,yVelocity))
    print(states)
'''
Identifying the start position
'''
def racerStart():
    for row in range(len(raceTrack)):
        for column in range(len(raceTrack)):
            if(raceTrack[row][column] == 'S'):
                state = (row,column)
                '''
                Appending Start points so that 
                we can choose random pints for start
                '''
                startStates.append(state)
                
'''
List of all possible actions for a particular state
'''
def actions(state):
    row,column = state
    if raceTrack[row][column] == 'F':
        return exit
    elif  raceTrack[row][column] == '#':
        #Function for previous position or start
        pass
    return possibleActions

'''
Providing Reward which is punihment in our world
'''
def giveRewards(state):
    row,column = state
    if raceTrack[row][column] == 'F':
        return 0
    return punishment


def stateTransitions(state,action):
    
    row, column = state
    if(raceTrack[row][column] == 'F'):
        return 100
    if(action == (0,0)):
        return state
    else:
        return[(0.8, validMove(state,possibleActions)),
               (0.2, validMove(state,[(0,0)]))]

def validMove(state,possibleActions):
    row, column = state
    for key in possibleActions:
        newRow, newColumn = key
        x = row+newRow
        y = column+newColumn
        if raceTrack[x][y] != '#':
            return True
        else:
            return False

def main():
    track = Track('R')
    track.show()
    print(track.start_positions)

if __name__ == "__main__":
    main()
