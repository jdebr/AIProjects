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

class ValueIteration():
    def __init__(self, epsilon, discount, track):
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
                self.Vtable[temp_state] = 0
        
    
    def get_reward(self, state):
        ''' Checks position of car and delivers appropriate reward '''
        if self.track.check_location(state[0], state[1]) == 'F':
            return 0
        else:
            return -1
    
    def valueIteration(self, epsilon , discount):
        print("*** Begin Value Iteration ***")
        #print("Threshold for convergence: " + str(epsilon))
        #print("Discount factor: " + str(discount))
        #print("Initializing Value Table to zero...")
        for Initialkey, InititalValue in self.Vtable.items():
            
            '''
            making a copy of states with value 0
            '''
            self.statesCopy[Initialkey] = 0
            #for action, value in InititalValue.items():
            #    self.statesCopy[Initialkey][action] = 0

        #discount = 0.1
        #print(self.statesCopy)
        self.sCopy = self.statesCopy.copy()
        V_current = self.statesCopy
        V_prev = self.sCopy
        converging = True
        conv_count = 0
        while converging:
            conv_count += 1
            print("Converging...iteration " + str(conv_count))
            converging = False
            if V_current is self.statesCopy:
                V_prev = self.statesCopy
                V_current = self.sCopy
            else:
                V_prev = self.sCopy
                V_current = self.statesCopy
            
            # Iterate all states
            for state in self.Vtable.keys():
#                 if state == (3, 20, 0, 0) and conv_count % 5 == 0:
#                     print("*** Example Temporal Difference Calculation ***")
#                     print(" ~ Current State: " + str(state))
                # Main calculation
                a_values = {}
                for action in self.possible_actions:
                    pos_states = self.stateTransitions(state, action)
                    # Calculate V[S] as sum over possible states 
                    v_sum = 0
                    for pstate in pos_states:
                        v_sum += pstate[0] * (self.get_reward(pstate[1]) + discount*V_prev[pstate[1]])
                    a_values[action] = v_sum
#                     if state == (3, 20, 0, 0) and conv_count % 5 == 0:
#                         print(" ~ Possible State Transitions for action " + str(action) +": " + str(pos_states))
#                         print(" ~ Calculated Value for action " + str(action) + ": " + str(v_sum))
                # Find action with max value and store it in current V
                best_value = max(a_values.values())
                V_current[state] = best_value
#                 if state == (3, 20, 0, 0) and conv_count % 5 == 0:
#                     print(" ~ Updating V-Table with max value: " + str(best_value))
                    
                # Check for convergence
                if abs(V_current[state]-V_prev[state]) > epsilon:
                    converging = True
        # Build Policy
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
                    v_sum += pstate[0] * (self.get_reward(pstate[1]) + discount*V_current[pstate[1]])
                argMax[action] = v_sum
            best_action = (0,0)
            best_value = argMax[best_action]
            for action, value in argMax.items():
                if value > best_value:
                    best_value = value 
                    best_action = action 
#             if printExample and state == (3, 20, 0, 0):
#                 printExample = False
#                 print("*** Example policy calculation ***")
#                 print(" ~ Finding best action for state " + str(state))
#                 print(" ~ Possible actions and resulting values for argMax calculation: " + str(argMax))
#                 print(" ~ Best action for state: " + str(state) + ": "+ str(best_action))
            # Assign Policy
            self.pi[state] = best_action
        print("Done")
    
    
    def stateTransitions(self,state,action):
        transitions = []
        
        for prob in [0.8, 0.2]:
            # Use simulator to return updated states
            self.agent.set_state(state)
            if prob > 0.5:
                # Apply action
                self.agent.set_acceleration(action[0], action[1])
            else:
                # set acceleration to 0
                self.agent.set_acceleration()
            # Move simulation deterministically
            self.agent.move_deterministic()
            # get new state
            new_state = self.agent.get_state()
            transitions.append((prob, new_state))
        return transitions
            

    def trial_run(self, max_moves=1000, show_track=False):
        ''' Attempts a trial run through the course, tracking total moves until the finish line is found or some max number is reached '''
        print("*** TRIAL RUN ***")
        num_moves = 0
        # Set agent at starting line
        start_state = self.track.get_random_start_state()
        self.agent.set_state(start_state)
        # Begin trial
        for i in range(max_moves):
            action = self.pi[self.agent.get_state()]
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
