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

80% we accelerate hence 0.8 
20% no acceleration hence 0.2
'''
possibleActions = [(1,0),(1,1),(0,1),(-1,0),(0,0),(-1,-1),(0,-1),(1,-1),(-1,1)]
raceTrack = []
states = defaultdict(list)
statesWall = []
startStates = []
punishment = -1
discount = 0.9
stateValue = {}
velocityX = 0
velocityY = 0
LoopCounter = 0
statesCopy = dict([])
sCopy = dict([])

def valueIteration(epsilon = 0.001):
    global LoopCounter
    
    '''
    calling the function to initialize the states dictionary
    '''
    racerStates()
    racerStart()
    print(len(states))
    for Initialkey, InititalValue in states.items():
        #print(key)
        '''
        making a copy of states with value 0
        '''
        statesCopy[Initialkey] = 0
    #discount = 0.1
    while True:
        #LoopCounter+=1
        sCopy = statesCopy.copy()
        #print(sCopy)
        delta = 0
        for key, value in states.items():
            LoopCounter+=1
            statesCopy[key] = giveRewards(key) + discount * max([sum([probability * sCopy[newState] for (probability, newState) in stateTransitions(key, a)])
                                    for a in actions(key)])
            delta = max(delta, abs(statesCopy[key] - sCopy[key]))
            #print("What" + str(delta))
            #print(statesCopy)
            #print(len(statesCopy))
            if delta < epsilon * (1-discount) / discount:
                print(sCopy)
                pi = policyIdentification(sCopy)
                return pi
            
def policyIdentification(sCopy):
    pi = {}
    for Initialkey, InititalValue in states.items():
        best_action = (0,0)
        for value in actions(Initialkey):
            if value > best_action:
                best_action = value
        a = best_action     
        pi[Initialkey] = sum([probability * sCopy[newState] for (probability, newState) in stateTransitions(Initialkey, a)])
    print(pi)
def utilityFunction(a,Initialkey,sCopy):
    return sum([probability * sCopy[newState] for (probability, newState) in stateTransitions(Initialkey, a)])

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
    return possibleActions

'''
Providing Reward which is punishment in our world
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
    tempVelocityX = velocityX
    tempVelocityY = velocityY
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
    '''    
    print("-----------")
    print(x,y)
    print(x+velocityX,y+velocityY)
    print((x+tempVelocityX,y+tempVelocityY))
    '''
    if((x + velocityX,y + velocityY) in states):
        if((x+tempVelocityX,y+tempVelocityY) in states):
            end = (x + velocityX, y + velocityY)
            crashed = check_for_crash((x , y), end)
            if not crashed:
                return[(0.8,(end[0],end[1])),(0.2,(x+tempVelocityX,y+tempVelocityY))]
            elif raceTrack[crashed[0]][crashed[1]] == 'F':
                return[(0.8,(crashed[0],crashed[1])),(0.2,(crashed[0],crashed[1]))]
            else:
                velocityX = 0
                velocityY = 0
                if True:
                    pos = random.choice(startStates)
                    return[(0.8,(pos[0],pos[1])),(0.2,(pos[0],pos[1]))]
                else:
                    return[(0.8,(crashed[0],crashed[1])),(0.2,(crashed[0],crashed[1]))]
        else:
            #i am confused here
            #return[(0.0,(x,y))]
            return[(0.8,(x,y)),(0.2,(x,y))]
    else:
        #i am confused here
        #return[(0.0,(x,y))]
        return[(0.8,(x,y)),(0.2,(x,y))]
    
    
def check_for_crash(start, end):
        ''' Method to check spaces in a line between two (x,y) locations on the track.
        If one of these spaces is a wall space, return the location the car would have
        been right before the crash.  Otherwise return false
        '''      
        x0 = start[0]
        x1 = end[0]
        y0 = start[1]
        y1 = end[1]
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x = x0
        y = y0
        n = 1 + dx + dy
        if x1 > x0 :
            x_inc = 1
        else :
            x_inc = -1
        if y1 > y0 :
            y_inc = 1
        else :
            y_inc = -1
        error = dx - dy
        dx *= 2
        dy *= 2
        oldX = x
        oldY = y

        for i in range(n, 0, -1):
            '''
            This needs to be checked it says x,y so what it should be
            '''
            if(raceTrack[x1][y1]) == 'F' :
                return (x1,y1)
            elif(raceTrack[x1][y1]) == '#' :
                return(oldX,oldY)
            else : 
                oldX = x 
                oldY = y
                if error > 0 :
                    x += x_inc
                    error -= dy
                else :
                    y += y_inc
                    error += dx
        return False 

'''      
A temporary function just for simplicity
'''

def fileReading():
    count = 0
    trackFile = 'D:/Shriyansh_PostGraduation/Artifical Intelligence/Project 4/O-track.txt'
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
    
def Qlearning_Otest():
    ''' Testing function for Q learning on O-Track'''
    q = QLearner(0.5, 0.9, 0.9,"O")
    q.track.show()
    q.train((3,4,0,1))
    q.train((4,3,0,1))
    q.train((20, 2, -1, 0))
    q.train((21, 4, -1, -1))
    q.train((20, 20, 0, -1))
    q.train((18, 22, 1, -1))
    q.train((4, 22, 1, 0))
    q.train((3, 20, 1, 1))
    q.train()
    for i in range(10):
        print(q.trial_run())
    
def Qlearning_Ltest():
    ''' Testing function for Q learning on L-track'''
    q = QLearner(0.5, 0.9, 0.9,"L")
    q.track.show()
    q.train((32, 2, 0, 1))
    q.train((32, 3, 0, 1))
    q.train()
    q.train()
    for i in range(10):
        print(q.trial_run())
    
def main():
    #crash_test()
    Qlearning_Otest()
    #Qlearning_Ltest()

if __name__ == "__main__":
    main()
