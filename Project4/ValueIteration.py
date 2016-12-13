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
import random


class ValueIteration():
    def __init__(self, discount, epsilon, track):
        self.discount = discount 
        self.epsilon = epsilon
        self.track = Track(track)
        self.agent = self.track.car 
        self.sCopy = {}
        self.pi = {}
        self.statesCopy = {}
        self.possible_actions = [(1,1),(1,0),(1,-1),(0,1),(0,0),(0,-1),(-1,1),(-1,0),(-1,-1)]
        self.possible_velocities = []
        for i in range(-5,6):
            for j in range(-5,6):
                self.possible_velocities.append((i,j))
        self.Vtable = {}
        # Initialize Q Table
        for pos in self.track.track_positions:
            for vel in self.possible_velocities:
                temp_state = (pos[0], pos[1], vel[0], vel[1])
                self.Vtable[temp_state] = 0
                #for action in self.possible_actions:
                #   self.Vtable[temp_state][action] = 0
        
    
    def get_reward(self, state):
        ''' Checks position of car and delivers appropriate reward '''
        if self.track.check_location(state[0], state[1]) == 'F':
            return 0
        else:
            return -1
    
    def valueIteration(self, epsilon , discount):
        for Initialkey, InititalValue in self.Vtable.items():
            '''
            making a copy of states with value 0
            '''
            self.statesCopy[Initialkey] = 0
            #for action, value in InititalValue.items():
            #    self.statesCopy[Initialkey][action] = 0
        #discount = 0.1
        print(len(self.statesCopy))       
        #V_current = self.statesCopy
        #V_prev = self.sCopy
        self.sCopy = self.statesCopy.copy()
        converging = True
        while converging:      
            #self.sCopy = self.statesCopy.copy()   
            delta = 0
            converging = False
            for key, value in self.Vtable.items():
                #for move, reward in value.items():
                actionValue = {}
                tempState = key
                for a in self.possible_actions:
                    newState = self.stateTransitions(key, a)
                    sumValue = 0
                    for pstate in newState:
                        tempState = pstate[1]
                        sumValue += pstate[0] * (self.get_reward(pstate[1]) + discount*self.sCopy[pstate[1]])
                    actionValue[a] = sumValue
                #print(tempState)    
                
                best_value = max(actionValue.values())
                self.statesCopy[key] = best_value
                
                #self.sCopy[key] = best_value
                # Check for convergence
                print(abs(self.statesCopy[key]-self.sCopy[key]))
                if abs(self.statesCopy[key]-self.sCopy[key]) > epsilon:
                    self.sCopy[key] = best_value
                    converging = True
        print()
        print("*** Value Iteration Complete, Building Action Policy***")
        printExample = True
        for state in self.Vtable.keys():
            # Argmax
            argMax = {}
            for action in self.possible_actions:
                pos_states = self.stateTransitions(state, action)
                # Calculate V[S] as sum over possible states 
                v_sum = 0
                for pstate in pos_states:
                    v_sum += pstate[0] * (self.get_reward(pstate[1]) + discount*self.sCopy[pstate[1]])
                argMax[action] = v_sum
            best_action = (0,0)
            best_value = argMax[best_action]
            for action, value in argMax.items():
                if value > best_value:
                    best_value = value 
                    best_action = action 
            if printExample and state == (3, 20, 0, 0):
                printExample = False
                print("*** Example policy calculation ***")
                print(" ~ Finding best action for state " + str(state))
                print(" ~ Possible actions and resulting values for argMax calculation: " + str(argMax))
                print(" ~ Best action for state: " + str(state) + ": "+ str(best_action))
            # Assign Policy
            self.pi[state] = best_action
        print("Done")
        print()
        '''
            self.statesCopy[key] = self.get_reward(key) + discount * max([sum([probability * self.sCopy[newState] for (probability, newState) in self.stateTransitions(key, a)])
                                                                        for a in self.possible_actions])
            
            delta = max(delta, abs(self.statesCopy[key] - self.sCopy[key]))
            self.sCopy[key] = self.statesCopy[key]
            print(delta)
            if  abs(self.statesCopy[key] - self.sCopy[key]) > (epsilon * (1-discount) / discount):
               #print(len(self.sCopy))
               #print(len(self.statesCopy))
                return delta
        '''
                
    #Not using for time being                
    def policyIdentification(self, sCopy):
        '''
        for Initialkey, InititalValue in self.Vtable.items():
            best_action = (0,0)
            best_value = self.sCopy[Initialkey]
            self.pi[Initialkey] = {}
            for a in self.possible_actions: 
                self.pi[Initialkey][a] = sum([probability * self.sCopy[newState] for (probability, newState) in self.stateTransitions(Initialkey, a)])
        print(self.pi)
        '''
    
        
    def stateTransitions(self,state,action):
        transitions = []
        for prob in [0.8 , 0.2]:
            self.agent.set_state(state)
            if prob > 0.5:
                self.agent.set_acceleration(action[0], action[1])
            else:
                self.agent.set_acceleration()
            self.agent.moveValueIteration()
            new_state = self.agent.get_state()
            transitions.append((prob,new_state))
        return transitions
        '''
        if(action == (0, 0)):
            return[(0.8,state),(0.2,state)]
        else:
            self.agent.set_state(state)
            self.agent.set_acceleration(action[0], action[1])
            returnValue = self.agent.moveValueIteration()
            return(returnValue[0],returnValue[1])
        
        #if(action == (0, 0)):
        #    return[(0.8,state),(0.2,state)]
        #else:
            
        self.agent.set_state(state)
        self.agent.set_acceleration(action[0], action[1])
        # Update state
        self.agent.move()
        new_state = self.agent.get_state()
        return[(0.8,new_state),(0.2,new_state)]
        #self.agent.set_state(state)
        self.agent.set_acceleration(action[0], action[1])
        # Update state
        new_Return = self.agent.moveValueIteration(state)
        #self.agent.set_state(new_Return[0][1])
        #print(new_Return[0][1])
        #print(new_Return[0])
        return[(new_Return[0]),(new_Return[1])]
        
        new_state = self.agent.get_state()
        self.agent.set_state(new_state)
        return[(0.8,new_state),(0.2,state)]
        '''
            
        
    def trial_run(self, max_moves=0):
        ''' Attempts a trial run through the course, tracking total moves until the finish line is found or some max number is reached '''
        print("*** TRIAL RUN ***")
        num_moves = 0
        self.valueIteration(self.epsilon, self.discount)
