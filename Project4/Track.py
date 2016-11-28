'''
CSCI 446 
Fall 2016
Project 4
Group 3

@author: Joe DeBruycker
@author: Shriyansh Kothari
@author: Sara Ounissi
'''

class Track():
    ''' Define a class for simulating a racetrack, which is specified by an external
    ASCII text file.  Contains methods for evaluating current position of a RaceCar object
    on the track.
    '''
    def __init__(self, trackShape):
        ''' Initialize track using text found in one of three files, specified by the trackshape
        which can either be 'R', 'O', or 'L'
        '''
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
            print("Size: " + str(line))
            # Remaining lines specify track
            line = f.readline()
            # Iterate rest of file, which is the track 
            while line:
                line = line.rstrip('\\n')
                trackLine = list(line)
                self.track.append(trackLine)
                line = f.readline()
                
        for line in self.track:
            for char in line:
                print(char, end="")