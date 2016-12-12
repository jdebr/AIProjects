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

class ValueIteration() :

    def __init__(self, track):

        self.track = Track(track)
        self.possibleActions = [(1,0),(1,1),(0,1),(-1,0),(0,0),(-1,-1),(0,-1),(1,-1),(-1,1)]
        self.states = defaultdict(list)
        self.statesWall = []
        self.startStates = []
        self.punishment = -1
        self.discount = 0.9
        slef.stateValue = {}
        self.velocityX = 0
        self.velocityY = 0
        self.LoopCounter = 0
        self.statesCopy = dict([])
        self.sCopy = dict([])
		


    def valueIteration(self,epsilon = 0.001):
    
        '''
        calling the function to initialize the states dictionary
        '''
        racerStates()
        racerStart()
        print(len(self.states))
        for Initialkey, InititalValue in self.states.items():
            #print(key)
            '''
            making a copy of states with value 0
            '''
            self.statesCopy[Initialkey] = 0
            #discount = 0.1
        while True:
            #LoopCounter+=1
            self.sCopy = self.statesCopy.copy()
            #print(sCopy)
            delta = 0
            for key, value in self.states.items():
                self.LoopCounter+=1
            self.statesCopy[key] = giveRewards(key) + self.discount * max([sum([probability * self.sCopy[newState] for (probability, newState) in stateTransitions(key, a)])
                                    for a in actions(key)])
            delta = max(delta, abs(self.statesCopy[key] - self.sCopy[key]))
            #print("What" + str(delta))
            #print(statesCopy)
            #print(len(statesCopy))
            if delta < epsilon * (1-self.discount) / self.discount:
                print(self.sCopy)
                pi = policyIdentification(self.sCopy)
                return pi
            
    def policyIdentification(self, sCopy):
        pi = {}
        for Initialkey, InititalValue in self.states.items():
            best_action = (0,0)
            for value in actions(Initialkey):
                if value > best_action:
                    best_action = value
            a = best_action     
            pi[Initialkey] = sum([probability * self.sCopy[newState] for (probability, newState) in stateTransitions(Initialkey, a)])
        print(pi)
	
    def utilityFunction(self,a,Initialkey,sCopy):
        return sum([probability * self.sCopy[newState] for (probability, newState) in stateTransitions(Initialkey, a)])

    '''
    Identify all possible states and appending to each state all the possible velocity it can have from -5 to +5 in both x and y
    '''    
    def racerStates(self):
        for row in range(len(self.track)):
            for column in range(len(self.track)):
                wallState = (row,column)
                self.statesWall.append(wallState)
                if(self.track[row][column] != '#'):
                    state = (row,column)
                    for xVelocity in range(-5,6):
                        for yVelocity in range(-5,6):
                            self.states[state].append((xVelocity,yVelocity))
    '''
    Identifying the start position
    '''
    def racerStart(self):
        for row in range(len(self.track)):
            for column in range(len(self.track)):
                if(self.track[row][column] == 'S'):
                    state = (row,column)
                    '''
                    Appending Start points so that 
                    we can choose random pints for start
                    '''
                    self.startStates.append(state)
                
    '''
    List of all possible actions for a particular state
    '''
    def actions(self,state):
        return self.possibleActions

    '''
    Providing Reward which is punishment in our world
    '''
    def giveRewards(self,state):
        row,column = state
        if self.track[row][column] == 'F':
            return 0
        return self.punishment

    def stateTransitions(self,state,action):
        x,y = state
        tempVelocityX = self.velocityX
        tempVelocityY = self.velocityY
        chance = random.random()
        probability = chance > 0.8
        if not probability:
            self.velocityX += action[0]
            self.velocityY += action[1]
        
        
        if self.velocityX > 5:
            self.velocityX = 5
        if self.velocityX < -5:
            self.velocityX = -5
        if self.velocityY > 5:
            self.velocityY = 5
        if self.velocityY < -5:
            self.velocityY = -5
        '''    
        print("-----------")
        print(x,y)
        print(x+velocityX,y+velocityY)
        print((x+tempVelocityX,y+tempVelocityY))
        '''
        if((x + self.velocityX,y + self.velocityY) in states):
            if((x+tempVelocityX,y+tempVelocityY) in states):
                end = (x + self.velocityX, y + self.velocityY)
                crashed = check_for_crash((x , y), end)
                if not crashed:
                    return[(0.8,(end[0],end[1])),(0.2,(x+tempVelocityX,y+tempVelocityY))]
                elif self.track[crashed[0]][crashed[1]] == 'F':
                    return[(0.8,(crashed[0],crashed[1])),(0.2,(crashed[0],crashed[1]))]
                else:
                    self.velocityX = 0
                    self.velocityY = 0
                    if True:
                        pos = random.choice(self.startStates)
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