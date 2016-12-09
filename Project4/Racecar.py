'''
CSCI 446 
Fall 2016
Project 4
Group 3

@author: Joe DeBruycker
@author: Shriyansh Kothari
@author: Sara Ounissi
'''
import random

class Racecar():
    ''' Defines a class representing a racecar agent which navigates the environment
    defined by a Track object.  Keeps track of current position as coordinates in the 
    track as well as the control variables for acceleration and the derived velocity
    along both axes.  
    '''
    # Store possible acceleration values for error checking
    possible_acc = [-1, 0, 1]
    
    def __init__(self, startX, startY, track):
        ''' Initializes a Racecar object at a starting position on a Track. '''
        # Associate a Track object
        self.track = track 
        # Set location
        self.x = startX 
        self.y = startY 
        # Initialize acceleration values as 0
        self.aX = 0
        self.aY = 0 
        # Initialize velocity values as 0
        self.vX = 0
        self.vY = 0
        
        
    def set_acceleration(self, accX = 0, accY = 0):
        ''' Sets the car's acceleration in the X and Y directions '''
        # Bounded as 0, -1, or 1 for each value
        if accX in Racecar.possible_acc and accY in Racecar.possible_acc:
            self.aX = accX 
            self.aY = accY 
        else:
            print("Invalid acceleration values")
            
    def get_state(self):
        ''' Returns tuple (x, y, x_velocity, y_velocity) '''
        state = (self.x, self.y, self.vX, self.vY)
        return state
    
    def set_state(self, state):
        ''' Manually sets the car's state '''
        self.x = state[0] 
        self.y = state[1]
        self.vX = state[2]
        self.vY = state[3]
        
    def check_location(self):
        ''' Returns ASCII char of car's current location'''
        return self.track.check_location(self.x, self.y)
        
    def move(self):
        ''' Applies the current values of the car's acceleration to the velocity 
        values (with a success rate of 80%), then attempts to update the car's 
        position based on the current velocity while checking for potential 
        collisions with the track walls
        '''
        # Generate random number between 0 and 1 to see if acceleration is successful
        chance = random.random()
        acc_fail = chance > 0.8 
        
        # Apply acceleration to velocity
        if not acc_fail:
            self.vX += self.aX
            self.vY += self.aY 
            
        # Bound velocity between -5 and 5
        if self.vX > 5:
            self.vX = 5
        if self.vX < -5:
            self.vX = -5
        if self.vY > 5:
            self.vY = 5
        if self.vY < -5:
            self.vY = -5
            
        # Check for collisions or finish
        end = (self.x + self.vX, self.y + self.vY)
        crash_or_finish = self.track.check_for_crash((self.x, self.y), end)
        
        # Update position
        if not crash_or_finish:
            self.x = end[0]
            self.y = end[1]
        elif self.track.check_location(crash_or_finish[0],crash_or_finish[1]) == 'F':
            # Cross the finish line
            self.x = crash_or_finish[0]
            self.y = crash_or_finish[1]
        else:
            # Crashed
            if self.track.restart:
                # Set to random starting position
                pos = random.choice(self.track.start_positions)
                self.x = pos[0]
                self.y = pos[1]
            else:
                # Set to value returned by check_for_crash()
                self.x = crash_or_finish[0]
                self.y = crash_or_finish[1]
        
        
        