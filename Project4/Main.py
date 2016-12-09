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
from QLearner import QLearner
from collections import defaultdict
from numpy import argmax
import random

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
statesWall = []
startStates = []
punishment = -1
discount = 0.4
stateValue = {}
velocityX = 0
velocityY = 0


def valueIteration(epsilon = 0.01):
    statesCopy = dict([])
    '''
    calling the function to initialize the states dictionary
    '''
    racerStates()
    for Initialkey, InititalValue in states.items():
        #print(key)
        '''
        making a copy of states with value 0
        '''
        statesCopy[Initialkey] = 0
    discount = 0.4
    while True:
        sCopy = statesCopy.copy()
        print(sCopy)
        delta = 0
        for key, value in states.items():
                statesCopy[key] = giveRewards(key) + discount * max([sum([probability * sCopy[newState] for (probability, newState) in stateTransitions(key, a)])
                                        for a in actions(key)])
                delta = max(delta, abs(statesCopy[key] - sCopy[key]))
                print(delta)
                if delta < epsilon * (1-discount) / discount:
                    return sCopy
                    
            
def policyIdentification(sCopy):
    pi = {}
    for Initialkey, InititalValue in states.items():
        pi[Initialkey] = argmax(actions(Initialkey), lambda a:utilityFunction(a,Initialkey,sCopy))
        
def utilityFunction(a,Initialkey,sCopy):
    return sum([p * sCopy[s1] for (p, s1) in stateTransitions(Initialkey, a)])

'''
Identify all possible states and appending to each state all the possible velocity it can have from -5 to +5 in both x and y
'''    
def racerStates():
    for row in range(len(raceTrack)):
        for column in range(len(raceTrack)):
            wallState = (row,column)
            statesWall.append(wallState)
            if(raceTrack[row][column] != '#'):
                state = (row,column)
                for xVelocity in range(-5,6):
                    for yVelocity in range(-5,6):
                        states[state].append((xVelocity,yVelocity))
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
    global velocityX
    global velocityY
    x,y = state
    failVelocityX = velocityX
    failVelocityY = velocityY
    chance = random.random()
    probability = chance > 0.8
    if not probability:
        velocityX += action[0]
        velocityY += action[1]
        
        
    if velocityX > 5:
        velocityX = 5
    if velocityX < -5:
        velocityX = -5
    if velocityY > 5:
        velocityY = 5
    if velocityY < -5:
        velocityY = -5
    if(raceTrack[x + velocityX][y + velocityY] in statesWall):
        end = raceTrack[x + velocityX][y + velocityY]
        if end == '.':
            newStateX = x + velocityX
            newStateY = y + velocityY
        elif end =='#':
            velocityX = 0
            velocityY = 0
            newStateX = x + velocityX
            newStateY = y + velocityY
            failVelocityX = velocityX
            failVelocityY = velocityY
            return [(0.8,(newStateX,newStateY)),(0.2,(x + failVelocityX,y + failVelocityY))]
    else:
        velocityX = 0
        velocityY = 0
        newStateX = x + velocityX
        newStateY = y + velocityY
        failVelocityX = velocityX
        failVelocityY = velocityY
        return [(0.8,(newStateX,newStateY)),(0.2,(x + failVelocityX,y + failVelocityY))]
    '''
    This is just rough draft for testing
    '''
    #print(newStateX,newStateY)
    

'''      
A temporary function just for simplicity
'''

def fileReading():
    count = 0
    trackFile = 'D:/Shriyansh_PostGraduation/Artifical Intelligence/Project 4/R-track.txt'
    with open(trackFile, 'r') as f:
            # First line specifies track dimensions
            line = f.readline()
            # Remaining lines specify track
            line = f.readline()
            while line:
                raceTrack.append([])
                line = line.rstrip('\n')
                for char in line:
                    raceTrack[count].append(char)
                count+=1
                line = f.readline()
                
def crash_test():
    ''' Testing function for crash detection in simulator '''
    q = QLearner(0.5, 0.9, "O")
    start = (3, 21)
    end = (7, 21)
    c = q.track.check_for_crash(start, end)
    if c:
        print("Crashed!")
        q.agent.set_state(c[0], c[1], 0, 0)
        q.track.show()
    else:
        print("Safe!")
        q.agent.set_state(end[0], end[1], 0, 0)
        q.track.show()
    #q.start()
    #q.track.show()
    
def main():
    crash_test()

if __name__ == "__main__":
    main()
