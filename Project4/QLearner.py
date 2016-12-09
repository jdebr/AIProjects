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

class QLearner():
    
    def __init__(self, discount, learning_rate, track):
        self.learning_rate = learning_rate
        self.discount = discount 
        self.track = Track(track)
        self.agent = self.track.car 
        self.Qtable = {}
        self.current_state = self.agent.get_state()
        self.possible_actions = [(1,1),(1,0),(1,-1),(0,1),(0,0),(0,-1),(-1,1),(-1,0),(-1,-1)]
        self.possible_velocities = []
        for i in range(-5,6):
            for j in range(-5,6):
                self.possible_velocities.append((i,j))
    
    def start(self):
        ''' Run the Q-learning algorithm '''
        # Initialize Q Table
        for pos in self.track.track_positions:
            for vel in self.possible_velocities:
                temp_state = (pos[0], pos[1], vel[0], vel[1])
                self.Qtable[temp_state] = {}
                for action in self.possible_actions:
                    self.Qtable[temp_state][action] = 0  
        # Main loop, when do we terminate?
        for i in range(1000):
            # E-greedy action selection, decaying epsilon?
            action = self.select_action(0.5)
            self.agent.set_acceleration(action[0], action[1])
            self.agent.move()
            new_state = self.agent.get_state()
            reward = self.get_reward(new_state)
            
            
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
                    
            return best_action
        
    def get_reward(self, state):
        ''' Checks position of car and delivers appropriate reward '''
        if self.track.check_location(state[0], state[1]) == 'F':
            return 0
        else:
            return -1
            
        