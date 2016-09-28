'''
CSCI 446 
Fall 2016
Project 2
Group 3

@author: Joe DeBruycker
@author: Shriyansh Kothari
@author: Sara Ounissi
'''

import random

'''
One Global Constant Variable for Wumpus World Size
'''

wumpusWorld = [5,10,15,20,25]

def worldCreation():
    worldMaker = []
    sizeSelected = random.choice(wumpusWorld)
    '''
    Print statements for Experimental and further design documents too
    '''
    print("Currently selected size of Wumpus World is " + str(sizeSelected) + " x " + str(sizeSelected))
    for i in range(sizeSelected):
        worldMaker.append([])
        for j in range(sizeSelected):
            worldMaker[i].append(j)
    print("2D Matrix is " + str(worldMaker))
    return worldMaker
    
    
    
def main():
    worldMatrix = worldCreation()
    
if __name__ == '__main__':
    main()
