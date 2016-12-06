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

possibleActions = [(1,0),(1,1),(0,1),(-1,0),(-1,-1),(0,-1),(0,0),(1,-1),(-1,1)]
raceTrack = []
states = []
startStates = []
punishment = -1

def value_Iteration():
    iterations = 100
    i = iterations
    while i > 0:
        pass
    
def racerStates():
    for row in range(len(raceTrack)):
        for column in range(len(raceTrack)):
            if(raceTrack[row][column] != '#'):
                state = (row,column)
                states.append(state) 

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

def actions(state):
    row,column = state
    if raceTrack[row][column] == 'F':
        return exit
    elif  raceTrack[row][column] == '#':
        pass
    return possibleActions

def giveRewards(state):
    row,column = state
    if raceTrack[row][column] == 'F':
        return 100
    return punishment

def main():
    track = Track('R')

if __name__ == "__main__":
    main()
