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
import random

class QLearner():
    
    def __init__(self, discount, learning_rate, track):
        self.learning_rate = learning_rate
        self.discount = discount 
        self.track = Track(track)
        self.agent = self.track.car 
        self.Q = {}
        self.current_state = self.agent.get_state()
        self.possible_actions = [(1,1),(1,0),(1,-1),(0,1),(0,0),(0,-1),(-1,1),(-1,0),(-1,-1)]
    
    def start(self):
        ''' Run the Q-learning algorithm '''
        
        for i in range(1000):
            # E-greedy action selection, decaying epsilon?
            action = self.select_action(0.9)
            
            
    def select_action(self, epsilon):
        ''' Selects one of the possible actions based on e-greedy strategy '''
        n = random.random()
        if n < epsilon:
            # Exploration
            return random.choice(self.possible_actions)
        else:
            # Exploitation
            best_action = random.choice(self.Q[self.current_state])
            for action in self.Q[self.current_state]:
                if self.Q[self.current_state][action] > self.Q[self.current_state][best_action]:
                    best_action = action
                    
        return best_action
            
        