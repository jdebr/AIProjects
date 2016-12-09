'''
CSCI 446 
Fall 2016
Project 4
Group 3

@author: Joe DeBruycker
@author: Shriyansh Kothari
@author: Sara Ounissi
'''
import Racecar
import random

class Track():
    ''' Define a class for simulating a racetrack, which is specified by an external
    ASCII text file.  Contains methods for evaluating current position of a RaceCar object
    on the track.
    '''
    def __init__(self, trackShape, car_position=None, restart_on_crash=False):
        ''' Initialize track using text found in one of three files, specified by the trackshape
        which can either be 'R', 'O', or 'L'.  Initializes a Racecar object at position defined by
        tuple car_position, or if not specified sets it to a random start location,
         and sets the restart penalty for car crashes.
        '''
        
        self.track = list()
        # List of lists of character representing the track
        
        self.start_positions = list()
        # List of tuples of (x,y) coordinates specifying potential starting locations
        
        self.track_positions = list()
        # List of tuples of (x,y) coordinates specifying potential starting locations
        
        self.restart = restart_on_crash
        # Whether car must restart after a crash
        
        # Read in specified track file
        trackFile = ''
        if trackShape == 'R':
            trackFile = 'R-track.txt'
        elif trackShape == 'O':
            trackFile = 'O-track.txt'
        elif trackShape == 'L':
            trackFile = 'L-track.txt'
        else: 
            print("Incorrect track selection, defaulting to O Track")
            trackFile = 'O-track.txt'
        
        # Parse and save track file
        with open(trackFile, 'r') as f:
            # First line specifies track dimensions
            line = f.readline()
            #print("Size: " + str(line))
            # Remaining lines specify track
            line = f.readline()
            # Iterate rest of file, which is the track 
            while line:
                line = line.rstrip('\\n')
                trackLine = list(line)
                self.track.append(trackLine)
                line = f.readline()
                
        # Find and save starting locations and track positions
        for i, line in enumerate(self.track):
            for j, char in enumerate(line):
                if char == 'S':
                    self.start_positions.append((j, len(self.track)-1-i))
                if char == '.' or char == 'S' or char == 'F':
                    self.track_positions.append((j, len(self.track)-1-i))
                
        # Initialize other class objects
        if not car_position:
            car_position = random.choice(self.start_positions)
        self.car = Racecar.Racecar(car_position[0], car_position[1], self)
                
    def show(self):
        ''' Method for printing ASCII representation of the state of the track '''
        for i, line in enumerate(self.track):
            for j, char in enumerate(line):
                if j == self.car.x and i == (len(self.track) - 1 - self.car.y):
                    # Print the car's location
                    print('R', end="")
                else:
                    print(char, end="")
                    
    def check_for_crash(self, start, end):
        ''' Method to check spaces in a line between two (x,y) locations on the track.
        If one of these spaces is a wall space, return the location the car would have
        been right before the crash.  Otherwise return false
        '''
        return False
        
        