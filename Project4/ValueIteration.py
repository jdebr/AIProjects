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
        self.current_state = self.agent.get_state()
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
                self.Vtable[temp_state] = {}
                for action in self.possible_actions:
                    self.Vtable[temp_state][action] = 0
        
    def select_action(self):
        # Exploitation
        best_action = (0,0)
        best_value = self.Vtable[self.current_state][best_action]
        for action, value in self.Vtable[self.current_state].items():
            if value > best_value:
                best_action = action
                best_value = value
                    
        return best_action
    
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
            self.statesCopy[Initialkey] = {}
            for action, value in InititalValue.items():
                self.statesCopy[Initialkey][action] = 0

        #discount = 0.1
        
        while True:
            #LoopCounter+=1
            self.sCopy = self.statesCopy.copy()
            
            #print(self.sCopy)
            delta = 0
            for key, value in self.Vtable.items():
                for move, reward in value.items():
                    self.statesCopy[key][move] = self.get_reward(key) + discount * max([sum([probability * self.sCopy[newState][a[0],a[1]] for (probability, newState) in self.stateTransitions(key, a)])
                                            for a in self.possible_actions])
                    delta = max(delta, abs(self.statesCopy[key][move] - self.sCopy[key][move]))
                    #print("What" + str(delta))
                    #print(statesCopy)
                    #print(len(statesCopy))
                    if delta < epsilon * (1-discount) / discount:
                        print(delta)
                        
                        print(self.sCopy)
                        carap = self.policyIdentification(self.sCopy)
                        return carap
                        
                
    def policyIdentification(self, self.sCopy):
        for Initialkey, InititalValue in self.Vtable.items():
            a = self.select_action()  
            self.pi[Initialkey][a] = sum([probability * self.sCopy[newState] for (probability, newState) in self.stateTransitions(Initialkey, a)])
        print(self.pi)
    
        
    def stateTransitions(self,state,action):
        
        print(state, action)
        self.agent.set_state(state)
        self.agent.set_acceleration(action[0], action[1])
            # Update state
        self.agent.move()
        #print(self.current_state)
        new_state = self.agent.get_state()
        print(new_state)
        return[(0.8,(new_state[0],new_state[1])),(0.2,(state[0],state[1]))]

    def trial_run(self, max_moves=0):
        ''' Attempts a trial run through the course, tracking total moves until the finish line is found or some max number is reached '''
        print("*** TRIAL RUN ***")
        num_moves = 0
        # Set agent at starting line
        start_state = self.track.get_random_start_state()
        self.agent.set_state(start_state)
        # Begin trial
        self.valueIteration(self.epsilon, self.discount)
        '''
        for i in range(max_moves):
            action = self.select_action()
            # Update car with action
            self.agent.set_acceleration(action[0], action[1])
            # Update state
            self.agent.move()
            self.current_state = self.agent.get_state()
            print(self.current_state)
            # Track score
            num_moves += 1
            # Show track
            #self.track.show()
            #print()
            #time.sleep(0.1)
            #x = input()
            # Terminate on finish
            if self.agent.check_location() == 'F':
                return num_moves
        return num_moves
        '''
