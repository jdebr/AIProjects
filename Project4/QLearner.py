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
    
    def __init__(self, discount, learning_rate, epsilon, track, restart_on_crash=False):
        self.learning_rate = learning_rate
        self.discount = discount 
        self.epsilon = epsilon
        self.track = Track(track, None, restart_on_crash)
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
    
    def train(self, start_state=None, learning_rate=None, discount=None, epsilon=None, iterations=1000000):
        ''' Run the Q-learning algorithm, potentially setting car's location to some other area and updating 
        learning rate and discount'''
        print("Training...")
        # Set state to restart learning if finish line is crossed during training
        if not start_state:
            self.agent.set_state(self.track.get_random_start_state())
        else:
            # Set up a group of starting states to train from
            starting_states = [(start_state)]
            starting_states.append((start_state[0]+1,start_state[1], start_state[2], start_state[3]))
            starting_states.append((start_state[0]-1,start_state[1], start_state[2], start_state[3]))
            starting_states.append((start_state[0],start_state[1]-1, start_state[2], start_state[3]))
            starting_states.append((start_state[0],start_state[1]+1, start_state[2], start_state[3]))
            self.agent.set_state(random.choice(starting_states))
        # Set other variables if passed in
        self.current_state = self.agent.get_state()
        if not learning_rate:
            learning_rate = self.learning_rate
        if not discount:
            discount = self.discount
        if not epsilon:
            epsilon = self.epsilon
        #print("Initial learning rate: " + str(learning_rate))
        #print("Initial discount: " + str(discount))
        #print("Initial e-greedy epsilon: " + str(epsilon))
                    
        # Main loop, when do we terminate?
        for i in range(iterations):
            # E-greedy action selection, decaying epsilon and learning rate
            #if i % (iterations/10) == 0:
            #    epsilon -= 0.05
            #    learning_rate -= 0.05
                #print()
                #print("*** Decaying epsilon and learning rate ***")
                #print(" ~ E-greedy epsilon decreased to " + str(epsilon))
                #print(" ~ Learning rate decreased to " + str(epsilon))
            action = self.select_action(epsilon)
            # Update car with action
            self.agent.set_acceleration(action[0], action[1])
            # Update state
            if not start_state:
                self.agent.move()
            else:
                self.agent.move(starting_states)
            old_state = self.current_state
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
                # Restart if at the finish
                if not start_state:
                    self.agent.set_state(self.track.get_random_start_state())
                else:
                    self.agent.set_state(random.choice(starting_states))
                self.current_state = self.agent.get_state()
            else:
                # Else keep training from new position
                self.current_state = new_state
            
#             if i == 50000:
#                 print()
#                 print("***EXAMPLE Q LEARNING CALCULATION***") 
#                 print(" ~ Current state (x, y, Vx, Vy): " + str(old_state)) 
#                 print(" ~ E-greedy action selection: " + str(action)) 
#                 print(" ~ New state: " + str(new_state))
#                 print(" ~ Current Q-Table value for state and action: " + str(Qsa))
#                 print(" ~ Reward: " + str(reward)) 
#                 print(" ~ Estimate of future reward: " + str(future_value))
#                 print(" ~ Updated Q-Table value for state and action: " + str(newQ))
            # Show track
            #self.track.show()
            #print()
            #time.sleep(0.1)
            #x = input()
                
        #print(self.Qtable)
        
    def trial_run(self, max_moves=10000, show_track=False):
        ''' Attempts a trial run through the course, tracking total moves until the finish line is found or some max number is reached '''
        print()
        print("*** TRIAL RUN ***")
        num_moves = 0
        # Set agent at starting line
        start_state = self.track.get_random_start_state()
        self.agent.set_state(start_state)
        # Begin trial
        for i in range(max_moves):
            action = self.select_action(0)
            # Update car with action
            self.agent.set_acceleration(action[0], action[1])
            # Update state
            self.agent.move()
            self.current_state = self.agent.get_state()
            # Track score
            num_moves += 1
            # Show track
            if show_track:
                print(" ~ Action selected from policy: " + str(action))
                self.track.show()
                print()
                #time.sleep(0.1)
                #x = input()
            # Terminate on finish
            if self.agent.check_location() == 'F':
                if show_track:
                    print("*** Finished course in " + str(num_moves) + " actions***")
                return num_moves
        return num_moves
            
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
            
        