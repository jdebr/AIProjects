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
import time

class QLearner():
    
    def __init__(self, discount, learning_rate, track):
        self.learning_rate = learning_rate
        self.discount = discount 
        self.track = Track(track)
        self.agent = self.track.car 
        self.current_state = self.agent.get_state()
        self.possible_actions = [(1,1),(1,0),(1,-1),(0,1),(0,0),(0,-1),(-1,1),(-1,0),(-1,-1)]
        self.possible_velocities = []
        for i in range(-5,6):
            for j in range(-5,6):
                self.possible_velocities.append((i,j))
        self.Qtable = {}
        # Initialize Q Table
        for pos in self.track.track_positions:
            for vel in self.possible_velocities:
                temp_state = (pos[0], pos[1], vel[0], vel[1])
                self.Qtable[temp_state] = {}
                for action in self.possible_actions:
                    self.Qtable[temp_state][action] = 0  
    
    def start(self, start_state=None, learning_rate=None, discount=None):
        ''' Run the Q-learning algorithm, potentially setting car's location to some other area and updating 
        learning rate and discount'''
        # Set state to restart learning if finish line is crossed during training
        if not start_state:
            start_state = self.track.get_random_start_state()
        else:
            self.agent.set_state(start_state)
        # Set other variables if passed in
        if not learning_rate:
            learning_rate = self.learning_rate
        if not discount:
            discount = self.discount
                    
        # Main loop, when do we terminate?
        for i in range(10000):
            # E-greedy action selection, decaying epsilon?
            action = self.select_action(0.5)
            self.agent.set_acceleration(action[0], action[1])
            self.agent.move()
            new_state = self.agent.get_state()
            reward = self.get_reward(new_state)
            # Update Q calculations
            Qsa = self.Qtable[self.current_state][action]
            #print("Q[s,a] = " + str(Qsa))
            # Estimated future reward
            future_value = self.get_max_future_value(new_state)
            # Update Q Table
            newQ = Qsa + (learning_rate * (reward + (discount * future_value) - Qsa))
            self.Qtable[self.current_state][action] = newQ
            # Update state
            if self.agent.check_location() == 'F':
                self.current_state = start_state
                self.agent.set_state(start_state)
            else:
                self.current_state = new_state
                
            # Show track
            self.track.show()
            print()
            time.sleep(0.1)
            x = input()
                
        #print(self.Qtable)
            
    def select_action(self, epsilon):
        ''' Selects one of the possible actions based on e-greedy strategy '''
        n = random.random()
        if n < epsilon:
            # Exploration
            return random.choice(self.possible_actions)
        else:
            # Exploitation
            best_action = (0,0)
            best_value = self.Qtable[self.current_state][best_action]
            for action, value in self.Qtable[self.current_state].items():
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
        
    def get_max_future_value(self, state):
        ''' Returns value from Q table for the action that maximizes value in given state '''
        best_action = (0,0)
        best_value = self.Qtable[state][best_action]
        for action, value in self.Qtable[state].items():
            if value > best_value:
                best_action = action
                best_value = value
        return best_value
            
        