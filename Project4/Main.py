'''
CSCI 446 
Fall 2016
Project 4
Group 3

@author: Joe DeBruycker
@author: Shriyansh Kothari
@author: Sara Ounissi
'''
from QLearner import QLearner
from ValueIteration import ValueIteration
import matplotlib.pyplot as plt
from statistics import mean
from statistics import median


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
    
def blah():
    q = QLearner(0.5,0.9,0.9,"O")
    q.train((3,4,0,1))
    q.train((4,3,0,1))
    q.train((20, 2, -1, 0))
    q.train((21, 4, -1, -1))
    q.train((20, 20, 0, -1))
    q.train((18, 22, 1, -1))
    q.train((4, 22, 1, 0))
    q.train((3, 20, 1, 1))
    q.train()
    q.train()
    for i in range(10):
        score = q.trial_run()
        print(score)
    
def Qlearning_Otest(iter=50000):
    ''' Testing function for Q learning on O-Track'''
    averages = []
    for k in range(20):
        averages.append([])
        
    for k in range(10):
        q = QLearner(0.5, 0.9, 0.9,"O")
        ep = 0.9
        lrate = 0.9
        scores = []
        #q.track.show()
        for j in range(20):
            scores.append([])
            q.train((3,4,0,1), learning_rate=lrate, epsilon=ep, iterations=iter)
            q.train((4,3,0,1), learning_rate=lrate, epsilon=ep, iterations=iter)
            q.train((20, 2, -1, 0), learning_rate=lrate, epsilon=ep, iterations=iter)
            q.train((21, 4, -1, -1), learning_rate=lrate, epsilon=ep, iterations=iter)
            q.train((20, 20, 0, -1), learning_rate=lrate, epsilon=ep, iterations=iter)
            q.train((18, 22, 1, -1), learning_rate=lrate, epsilon=ep, iterations=iter)
            q.train((4, 22, 1, 0), learning_rate=lrate, epsilon=ep, iterations=iter)
            q.train((3, 20, 1, 1), learning_rate=lrate, epsilon=ep, iterations=iter)
            q.train(learning_rate=lrate, epsilon=ep, iterations=iter)
            q.train(learning_rate=lrate, epsilon=ep, iterations=iter)
            for i in range(10):
                score = q.trial_run()
                scores[j].append(score)
            ep -= 0.02
            lrate -= 0.02
                
        z = [mean(l) for l in scores]
        for j in range(len(z)):
            averages[j].append(z[j])
        
    print(averages)
    x = [e*iter for e in range(20)]
    y = [median(l) for l in averages]
    print(y)
    plt.plot(x, y, marker='o', linestyle='-', color='r')
    plt.ylabel('Score')
    plt.xlabel('Training Samples Per Location')
    plt.show()
    
def Qlearning_Ltest(iter=30000):
    ''' Testing function for Q learning on L-track'''
    averages = []
    for k in range(20):
        averages.append([])
        
    for k in range(10):
        q = QLearner(0.5, 0.9, 0.9,"L")
        ep = 0.9
        lrate = 0.9
        scores = []
        #q.track.show()
        for j in range(20):
            scores.append([])
            #q.track.show()
            q.train((32, 2, 0, 1), learning_rate=lrate, epsilon=ep, iterations=iter)
            q.train((32, 3, 0, 1), learning_rate=lrate, epsilon=ep, iterations=iter)
            q.train(learning_rate=lrate, epsilon=ep, iterations=iter)
            q.train(learning_rate=lrate, epsilon=ep, iterations=iter)
            for i in range(10):
                score = q.trial_run()
                scores[j].append(score)
            ep -= 0.02
            lrate -= 0.02
            
        z = [mean(l) for l in scores]
        for j in range(len(z)):
            averages[j].append(z[j])
        
    print(averages)
    x = [e*iter for e in range(20)]
    y = [median(l) for l in averages]
    print(y)
    plt.plot(x, y, marker='o', linestyle='-', color='r')
    plt.ylabel('Score')
    plt.xlabel('Training Samples Per Location')
    plt.show()
        
def Qlearning_Rtest(iter=100000):
    ''' Testing function for Q learning on R-track'''
    averages = []
    for k in range(20):
        averages.append([])
        
    for k in range(10):
        q = QLearner(0.5, 0.9, 0.9, "R")
        ep = 0.9
        lrate = 0.9
        scores = []
        #q.track.show()
        for j in range(20):
            scores.append([])
            q.train((23,11, 1, -1), learning_rate=lrate, epsilon=ep, iterations=iter)
            q.train((12,17, 1, -1), learning_rate=lrate, epsilon=ep, iterations=iter)
            q.train((13,17, 1, -1), learning_rate=lrate, epsilon=ep, iterations=iter)
            q.train((21,25, 0, -1), learning_rate=lrate, epsilon=ep, iterations=iter)
            q.train((21,23, -1, -1), learning_rate=lrate, epsilon=ep, iterations=iter)
            q.train((5, 24, 1, 0), learning_rate=lrate, epsilon=ep, iterations=iter)
            q.train((4, 23, 1, 1), learning_rate=lrate, epsilon=ep, iterations=iter)
            q.train(learning_rate=lrate, epsilon=ep, iterations=iter)
            q.train(learning_rate=lrate, epsilon=ep, iterations=iter)
            for i in range(10):
                score = q.trial_run()
                scores[j].append(score)
            ep -= 0.02
            lrate -= 0.02
            
        z = [mean(l) for l in scores]
        for j in range(len(z)):
            averages[j].append(z[j])
        
    print(averages)
    x = [e*iter for e in range(20)]
    y = [median(l) for l in averages]
    print(y)
    plt.plot(x, y, marker='o', linestyle='-', color='r')
    plt.ylabel('Score')
    plt.xlabel('Training Samples Per Location')
    plt.show()
        
def Qlearning_Rtest_withReset(iter=500000):
    ''' Testing function for Q learning on R-track'''
    averages = []
    for k in range(20):
        averages.append([])
        
    for k in range(10):
        q = QLearner(0.5, 0.9, 0.9, "R", True)
        ep = 0.9
        lrate = 0.9
        scores = []
        #q.track.show()
        for j in range(20):
            scores.append([])
            q.train((23,11, 1, -1), learning_rate=lrate, epsilon=ep, iterations=iter)
            q.train((12,17, 1, -1), learning_rate=lrate, epsilon=ep, iterations=iter)
            q.train((13,17, 1, -1), learning_rate=lrate, epsilon=ep, iterations=iter)
            q.train((21,25, 0, -1), learning_rate=lrate, epsilon=ep, iterations=iter)
            q.train((21,23, -1, -1), learning_rate=lrate, epsilon=ep, iterations=iter)
            q.train((5, 24, 1, 0), learning_rate=lrate, epsilon=ep, iterations=iter)
            q.train((4, 23, 1, 1), learning_rate=lrate, epsilon=ep, iterations=iter)
            q.train(learning_rate=lrate, epsilon=ep, iterations=iter)
            q.train(learning_rate=lrate, epsilon=ep, iterations=iter)
            for i in range(10):
                score = q.trial_run()
                scores[j].append(score)
            ep -= 0.02
            lrate -= 0.02
            
        z = [mean(l) for l in scores]
        for j in range(len(z)):
            averages[j].append(z[j])
        
    print(averages)
    x = [e*iter for e in range(20)]
    y = [median(l) for l in averages]
    print(y)
    plt.plot(x, y, marker='o', linestyle='-', color='r')
    plt.ylabel('Score')
    plt.xlabel('Training Samples Per Location')
    plt.show()

def ValueIteration_Otest():
    VI = ValueIteration(0.00000000001, 0.5, "O")
    VI.valueIteration(0.00000000001,0.5)
    for i in range(10):
        print(VI.trial_run())

def ValueIteration_Ltest():
    VI = ValueIteration(0.00000000001, 0.5, "L")
    VI.valueIteration(0.00000000001,0.5)
    for i in range(10):
        print(VI.trial_run())
        
def ValueIteration_Rtest():
    VI = ValueIteration(0.00000000001, 0.5, "R")
    VI.valueIteration(0.00000000001,0.5)
    for i in range(10):
        print(VI.trial_run())

def VI_R_reset():
    VI = ValueIteration(0.00000000001, 0.5, "R", restart=True)
    VI.valueIteration(0.00000000001, 0.5)
    for i in range(10):
        print(VI.trial_run())
    
def main():
    #blah()
    #Qlearning_Otest()
    #Qlearning_Ltest()
    Qlearning_Rtest()
    #Qlearning_Rtest_withReset()

if __name__ == "__main__":
    main()
