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

class Track():
    ''' Define a class for simulating a racetrack, which is specified by an external
    ASCII text file.  Contains methods for evaluating current position of a RaceCar object
    on the track.
    '''
    def __init__(self, trackShape, carX, carY):
        ''' Initialize track using text found in one of three files, specified by the trackshape
        which can either be 'R', 'O', or 'L'.  Initializes a Racecar object at position defined by
        carX and carY
        '''
        # Self.track is a list of lists of character representing the track
        self.track = list()
        
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
                
        # Initialize other class objects
        self.car = Racecar.Racecar(carX, carY)
                
    def show(self):
        ''' Method for printing ASCII representation of the state of the track '''
        for i, line in enumerate(self.track):
            for j, char in enumerate(line):
                if j == self.car.x and i == (len(self.track) - 1 - self.car.y):
                    # Print the car's location
                    print('R', end="")
                else:
                    print(char, end="")
        
        