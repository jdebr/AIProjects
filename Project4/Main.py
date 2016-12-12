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
from QLearner import QLearner
from ValueIteration import ValueIteration
from collections import defaultdict
from numpy import argmax
import random

'''
Things to keep in mind

accelerationList = [-1,0,1]
punishment = -1
racer_State = {}
racer_Reward = {}

80% we accelerate hence 0.8 
20% no acceleration hence 0.2
'''

def crash_test():
    ''' Testing function for crash detection in simulator '''
    q = QLearner(0.5, 0.9, "O")
    start = (3, 21)
    end = (7, 21)
    c = q.track.check_for_crash(start, end)
    if c:
        print("Crashed!")
        q.agent.set_state(c[0], c[1], 0, 0)
        q.track.show()
    else:
        print("Safe!")
        q.agent.set_state(end[0], end[1], 0, 0)
        q.track.show()
    #q.start()
    #q.track.show()
    
def Qlearning_Otest():
    ''' Testing function for Q learning on O-Track'''
    q = QLearner(0.5, 0.9, 0.9,"O")
    q.track.show()
    q.train((3,4,0,1))
    q.train((4,3,0,1))
    q.train((20, 2, -1, 0))
    q.train((21, 4, -1, -1))
    q.train((20, 20, 0, -1))
    q.train((18, 22, 1, -1))
    q.train((4, 22, 1, 0))
    q.train((3, 20, 1, 1))
    q.train()
    for i in range(10):
        print(q.trial_run())
    
def Qlearning_Ltest():
    ''' Testing function for Q learning on L-track'''
    q = QLearner(0.5, 0.9, 0.9,"L")
    q.track.show()
    q.train((32, 2, 0, 1))
    q.train((32, 3, 0, 1))
    q.train()
    q.train()
    for i in range(10):
        print(q.trial_run())
		
        
def Qlearning_Rtest():
    ''' Testing function for Q learning on R-track'''
    q = QLearner(0.5, 0.9, 0.9, "R", True)
    q.track.show()
    q.train((23,11, 1, -1))
    q.train((12,17, 1, -1))
    q.train((13,17, 1, -1))
    q.train((21,25, 0, -1))
    q.train((21,23, -1, -1))
    q.train((5, 24, 1, 0))
    q.train((4, 23, 1, 1))
    q.train()
    q.train()
    for i in range(10):
        print(q.trial_run())

def ValueIteration_Otest():
    VI = ValueIteration(0.001, 0.9, "O")
    VI.track.show()
    VI.valueIteration(0.001,0.9)
    VI.track.show()
	#to do

def ValueIteration_Ltest():
    pass 
    
def main():
    #crash_test()
    #Qlearning_Otest()
    #Qlearning_Ltest()
	ValueIteration_Otest()
    #Qlearning_Rtest()

if __name__ == "__main__":
    main()
