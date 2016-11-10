'''
CSCI 446 
Fall 2016
Project 3
Group 3

@author: Joe DeBruycker
@author: Shriyansh Kothari
@author: Sara Ounissi
'''

import os
import random
import math
import TreeNode


def readingFiles():
    
    #Change the directory where you are storing the data.
    fileNames = os.listdir("D:/Shriyansh_PostGraduation/Artifical Intelligence/Project 3")
    print(fileNames)
    for name in fileNames:
        if "data.txt" in name:
            fileOpen = open("D:/Shriyansh_PostGraduation/Artifical Intelligence/Project 3/" + str(name))
            for lines in fileOpen:
                '''
                #Just Checking if reading individual elements in the line selected.
                for element in lines:
                    print(element)
                '''
                print(lines.rstrip())
            fileOpen.close()
            
            
            
def readFile(fileName):
    ''' Reads in the file specified and parses the content into a list of lists,
    each individual list containing string items representing the text data
    '''
    dataSet = list()
    
    with open(fileName, 'r') as f:
        # split raw text on new lines
        txtData = f.read().splitlines()
        # parse out empty lines and split individual values within data instances
        for line in txtData:
            if not line:
                continue
            datum = line.split(',')
            dataSet.append(datum)
            
    return dataSet

def discretize(data, numBins):
    ''' Takes dataset of continuous features and applies equal width binning 
    to discretize each feature to an integer value numBins
    '''
    # Loop for each feature
    for i in range(len(data[0])):
        # Find min and max feature values
        featMin = data[0][i]
        featMax = data[0][i]
        for line in data:
            if line[i] < featMin:
                featMin = line[i]
            if line[i] > featMax:
                featMax = line[i]
                
        # Find bin width and bin features
        binWidth = (featMax - featMin) / numBins
        for j in range(len(data)):
            feat = data[j][i]
            feat = feat - featMin 
            # Check each bin to see if feature falls within
            for k in range(1, numBins+1):
                if feat - (k * binWidth) <= 0:
                    data[j][i] = k
                    break
                
    return data

def imputeVote(party, feature):
    ''' Returns a boolean number representing vote data based on the greatest likelihood
    for that feature given the party (class label).  Uses domain knowledge from the metadata
    to determine this value.
    '''
    if party == 'democrat':
        if feature == 1:
            return 1
        elif feature == 2:
            return random.choice([0,1])
        elif feature == 3:
            return 1
        elif feature == 4:
            return 0
        elif feature == 5:
            return 0
        elif feature == 6:
            return 0
        elif feature == 7:
            return 1
        elif feature == 8:
            return 1
        elif feature == 9:
            return 1
        elif feature == 10:
            return 0
        elif feature == 11:
            return 1
        elif feature == 12:
            return 0
        elif feature == 13:
            return 0
        elif feature == 14:
            return 0
        elif feature == 15:
            return 1
        elif feature == 16:
            return 1
    else:
        if feature == 1:
            return 0
        elif feature == 2:
            return random.choice([0,1])
        elif feature == 3:
            return 0
        elif feature == 4:
            return 1
        elif feature == 5:
            return 1
        elif feature == 6:
            return 1
        elif feature == 7:
            return 0
        elif feature == 8:
            return 0
        elif feature == 9:
            return 0
        elif feature == 10:
            return 1
        elif feature == 11:
            return 0
        elif feature == 12:
            return 1
        elif feature == 13:
            return 1
        elif feature == 14:
            return 1
        elif feature == 15:
            return 0
        elif feature == 16:
            return 1
            
def getIris():
    ''' Processes the Iris data, returning a tuple of 2 lists: ([data],[labels]),
    where the data has been turned into lists of numerical discretized data and 
    class labels are strings 
    '''
    iris = readFile("../data/iris.data")
    data = list()
    labels = list()
    
    for item in iris:
        data.append([float(item[0]), float(item[1]), float(item[2]), float(item[3])])
        labels.append(item[4])
        
    discretize(data, 10)
    
    return (data, labels)

def getGlass():
    ''' Processes the Glass data, returning a tuple of 2 lists: ([data],[labels]),
    where the data has been turned into lists of numerical discretized data and 
    class labels are strings 
    '''
    glass = readFile("../data/glass.data")
    data = list()
    labels = list()
    
    # Glass has 11 attributes, the first being an ID and the last being the Class
    for item in glass:
        data.append([float(item[1]), float(item[2]), float(item[3]), float(item[4]), float(item[5]), float(item[6]), float(item[7]), float(item[8]), float(item[9])])
        labels.append(item[10])
    
    discretize(data, 10)
    
    return (data, labels)

def getSoybean():
    ''' Processes the Soybean data, returning a tuple of 2 lists: ([data],[labels]),
    where the data has been turned into lists of numerical discretized data and 
    class labels are strings 
    '''
    soybean = readFile("../data/soybean-small.data")
    data = list()
    labels = list()
    
    # Soybeans has 36 attributes,  the last being the Class
    for item in soybean:
        newData = list()
        for i in range(len(item)-1):
            newData.append(int(item[i]))
        data.append(newData)
        labels.append(item[-1])
    
    return (data, labels)

def getBreastCancer():
    ''' Processes the Breast Cancer data, returning a tuple of 2 lists: ([data],[labels]),
    where the data has been turned into lists of numerical discretized data and 
    class labels are strings.  Data with missing values are omitted from final dataset.
    '''
    cancer = readFile("../data/breast-cancer-wisconsin.data")
    data = list()
    labels = list()
    
    # Breast Cancer has 11 attributes, the first being an ID and the last being the Class
    for item in cancer:
        # Check for missing feature, ignore data instance if '?' exists
        omit = False
        for feature in item:
            if feature == '?':
                omit = True
                break
        if not omit:    
            data.append([int(item[1]), int(item[2]), int(item[3]), int(item[4]), int(item[5]), int(item[6]), int(item[7]), int(item[8]), int(item[9])])
            labels.append(item[10])
    
    return (data, labels)

def getVote():
    ''' Processes the Vote data, returning a tuple of 2 lists: ([data],[labels]),
    where the data has been turned into lists of numerical boolean data and 
    class labels are strings. Missing values are imputed to the most likely boolean
    value given the class
    '''
    vote = readFile("../data/house-votes-84.data")
    data = list()
    labels = list()
    
    # Vote has 17 attributes,  the first being the Class
    for item in vote:
        newData = list()
        for i in range(1, len(item)):
            # Encode data as 'y' -> 1, 'n'-> 0
            if item[i] == 'y':
                newData.append(1)
            elif item[i] == 'n':
                newData.append(0)
            else:
                # Impute missing values based on class likelihood
                newData.append(imputeVote(item[0],i))
        data.append(newData)
        labels.append(item[0])
    
    return (data, labels)

def calculateEntropy(dataSet, labels):
    ''' Returns the calculation of the entropy for a dataSet as the sum
    of the probability of each class times the log probability of that class
    '''
    # Count numbers of each class and total number of classes
    classCounts = list()
    localCount = 0
    currentLabel = labels[0]
    for label in labels:
        # Count instances of each class, assumes class labels are grouped
        if label == currentLabel:
            localCount += 1
        # Count total number of classes
        else:
            classCounts.append(localCount)
            localCount = 1
            currentLabel = label
    # Append final localcount
    classCounts.append(localCount)
    
    # Entropy Calculation
    entropy = 0.0
    for i in range(len(classCounts)):
        pClass = float(classCounts[i])/len(dataSet)
        entropy += (pClass * math.log2(pClass))
    
    return -entropy

def calculateGain(feature, dataSet, labels):
    ''' Calculate the information gain as the difference in entropy from before
    to after the dataset is split on feature
    '''
    entropy = calculateEntropy(dataSet, labels)
    newEntropy = 0.0
    
    # Collect all possible values for feature
    featValues = set()
    for data in dataSet:
        featValues.add(data[feature])
    #print("Num Splits: " + str(len(featValues)))
        
    # For each value feature can take on measure entropy of that subset
    for value in featValues:
        subData = list()
        subLabels = list()
        # Get subset
        for i in range(len(dataSet)):
            if dataSet[i][feature] == value:
                subData.append(dataSet[i])
                subLabels.append(labels[i])
        #print("Subdata: " + str(subData))
        #print("Sublabels: " + str(subLabels))
        # Get subEntropy, multiply by probability, and add to newEntropy
        subEntropy = calculateEntropy(subData, subLabels)
        p = float(len(subData))/float(len(dataSet))
        newEntropy += (subEntropy * p)
        #print("Subentropy: " + str(subEntropy))
        #print("New Entropy: " + str(newEntropy))
        
    # Calculate gain
    gain = entropy - newEntropy
    return gain

'''
Naive Bayes 
'''

def classSeperation(dataValues):
    classDictionary = {}
    #print(len(dataValues))
    tempSet = dataValues[-1]
    #print(tempSet)
    #print(dataValues[0][1])
    for value in range(len(dataValues[0])):
        #print(dataValues[0][value])
        if (tempSet[value] not in classDictionary):
            #print(classDictionary)
            classDictionary[tempSet[value]] = [] 
        classDictionary[tempSet[value]].append(dataValues[0][value])

    #print(len(classDictionary))
    print(classDictionary)
    #attributeCount(classDictionary)
    

def main():
    irisData = getIris()
    gain = calculateGain(2,irisData[0], irisData[1])
    print("gain: " + str(gain))
    
if __name__ == '__main__':
    main()
