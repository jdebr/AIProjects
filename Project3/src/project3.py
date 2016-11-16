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
import operator
from nltk.chunk.util import accuracy

conditionalProbabilityValue = {}
totalClassValue = {}
classProbabilityValue = {}

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

# I WILL REMOVE THIS PER DESIGN DOCUMENT FEEDBACK...joe
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
    classCounts = {}
    for label in labels:
        # Count instances of each class
        if label in classCounts:
            classCounts[label] += 1
        # Count total number of classes
        else:
            classCounts[label] = 1
    #print(classCounts)
    
    # Entropy Calculation
    entropy = 0.0
    for count in classCounts.values():
        pClass = float(count)/len(dataSet)
        entropy += (pClass * math.log2(pClass))
    
    return -entropy

def calculateGainRatio(feature, dataSet, labels):
    ''' Calculate the information gain as the difference in entropy from before
    to after the dataset is split on feature, calculate the intrinsic value of that feature,
    then return the Gain Ratio as G/IV
    '''
    entropy = calculateEntropy(dataSet, labels)
    expectedEntropy = 0.0
    intrinsicValue = 0.0
    
    # Collect all possible values for feature
    featValues = set()
    for data in dataSet:
        featValues.add(data[feature])
    #print("Num Splits: " + str(len(featValues)))
        
    # For each value feature can take on measure entropy of that subset
    for value in featValues:
        subData = list()
        subLabels = list()
        # Get subset of data for that value
        for i in range(len(dataSet)):
            if dataSet[i][feature] == value:
                subData.append(dataSet[i])
                subLabels.append(labels[i])
        #print("Subdata: " + str(subData))
        #print("Sublabels: " + str(subLabels))
        
        # Get subEntropy, multiply by probability, and add to newEntropy
        subEntropy = calculateEntropy(subData, subLabels)
        p = float(len(subData))/float(len(dataSet))
        expectedEntropy += (subEntropy * p)
        intrinsicValue += (p * math.log2(p))
        #print("Subentropy: " + str(subEntropy))
        
        
    # Calculate gain
    gain = entropy - expectedEntropy
    #print("Gain: " + str(gain))
    #print("Expected Entropy: " + str(expectedEntropy))
    #print("Intrinsic Value: " + str(-intrinsicValue))
    
    # Calculate gain ratio, watch out for divide by zero!
    if int(intrinsicValue) == 0:
        return gain
    else:
        return (gain/-intrinsicValue)

def run_ID3(trainData, trainLabels, testData, testLabels, validationData, validationLabels):
    ''' Run the ID3 algorithm, building a decision tree using the trainData and trainLabels
    and then testing the tree using the testData and testLabels
    '''
    # Build initial set of feature indices for creating decision tree
    featureIndices = list()
    for i in range(len(trainData[0])):
        featureIndices.append(i)
        
    # Build decision tree
    ID3 = build_ID3(trainData, trainLabels, featureIndices)
    
    # Prune 
    prune(ID3, ID3, validationData, validationLabels)
    
    # Test decision tree using testData, return classification accuracy
    total = float(len(testData))
    numCorrect = 0
    for i in range(len(testData)):
        prediction = ID3.test(testData[i])
        if testLabels[i] == prediction:
            numCorrect += 1
        #print("ID3 Prediction: " + prediction)
        #print("Actual Class: " + testLabels[i])
            
    accuracy = numCorrect/total
    print("Number correctly classified: " + str(numCorrect))
    print("ID3 Accuracy: " + str(accuracy))

def build_ID3(trainData, trainLabels, features):
    ''' Build a node of the decision tree for ID3 by choosing the attribute from features that
    results in the highest gain ratio when splitting the trainData and creating child nodes for 
    all possible values of that attribute.  If features is empty, return the majority label of the 
    trainLabels.  If all trainLabels are the same, return that label.
    '''
    root = TreeNode.TreeNode()
    
    # If all trainLabels are the same return the node with that label
    nodeLabel = trainLabels[0]
    allSame = True
    for label in trainLabels:
        if label != nodeLabel:
            allSame = False
            break
    if allSame:
        root.setLabel(nodeLabel)
        return root
    
    # If no features remain for testing then return node with label of the majority class label
    if not features:
        majorityLabel = trainLabels[0]
        currentLabel = trainLabels[0]
        majorityCount = 1
        currentCount = 0
        for label in trainLabels:
            if label == currentLabel:
                currentCount += 1
            else:
                if currentCount > majorityCount:
                    majorityLabel = currentLabel
                    majorityCount = currentCount
                currentLabel = label 
                currentCount = 1
        root.setLabel(majorityLabel)
        return root
    
    # Otherwise find feature with highest gain ratio and split dataset by creating children nodes
    bestFeature = features[0]
    bestGR = calculateGainRatio(bestFeature, trainData, trainLabels)
    for feature in features:
        currentGR = calculateGainRatio(feature, trainData, trainLabels)
        if currentGR > bestGR:
            bestFeature = feature
            bestGR = currentGR
    root.setFeature(bestFeature)
    # Find possible values of best feature for splitting dataset
    featValues = set()
    for data in trainData:
        featValues.add(data[bestFeature])
    for value in featValues:
        # Get subset of dataset and labels with that feature value and create a child node using that subset
        subData = list()
        subLabels = list()
        subFeatures = features.copy()
        for i in range(len(trainData)):
            if trainData[i][bestFeature] == value:
                subData.append(trainData[i])
                subLabels.append(trainLabels[i])
        subFeatures.remove(bestFeature)
        child = build_ID3(subData, subLabels, subFeatures)
        root.addChild(child, value)
    # Set a pruning label as the majority label for classes at this point in the tree
    classCounts = {}
    for label in trainLabels:
        # Count instances of each class
        if label in classCounts:
            classCounts[label] += 1
        # Count total number of classes
        else:
            classCounts[label] = 1
    root.setPruneLabel(max(classCounts))
    return root
        
def prune(tree, node, data, labels):
    ''' Use a validation set to test the accuracy of the tree before and after reduced error pruning,
    which is done by iterating through internal tree nodes and setting their labels to be the majority
    label of their children.  If a pruned tree performs better than an unpruned tree, keep the pruned
    tree.
    '''
    # Base case
    if node.label:
        return
    # Start with trying to prune child nodes
    for child in node.children:
        prune(tree, child, data, labels)
    # Get unpruned accuracy
    total = float(len(data))
    numCorrect = 0
    for i in range(len(data)):
        prediction = tree.test(data[i])
        if labels[i] == prediction:
            numCorrect += 1
    accuracy = numCorrect/total
    # Prune self by setting label to majority class from training data and test
    node.setLabel(node.prune_label)
    numCorrect = 0
    for i in range(len(data)):
        prediction = tree.test(data[i])
        if labels[i] == prediction:
            numCorrect += 1      
    new_accuracy = numCorrect/total
    print("Pruned accuracy: " + str(new_accuracy))
    # Set label back to none if this results in decreased accuracy
    if new_accuracy < accuracy:
        node.label = None
    # Otherwise keep the pruned tree and set the best accuracy
    else:
        print("Node pruned!")
    return

'''
Naive Bayes 
'''

def classSeperation(trainData, trainLabels, testData, testLabels):
    '''
    In this block we join the training Labels and Data into one class dictionary
    '''
    classDictionary = {}
    for value in range(len(trainData)):
        if (trainLabels[value] not in classDictionary):
            classDictionary[trainLabels[value]] = [] 
        classDictionary[trainLabels[value]].append(trainData[value])

    print("Class Dictionary after joining training labels and data")
    print(classDictionary)
    attributeCount(classDictionary)
    
    # Test Naive Bayes using testData
    totalTest = float(len(testData))
    print("Print total test length " + str(totalTest))
    numCorrect = 0
    for i in range(len(testData)):
        prediction = testingNaiveBayes(testData[i])
        if testLabels[i] == prediction:
            numCorrect += 1
            
    accuracy = numCorrect/totalTest
    print("Number correctly classified: " + str(numCorrect))
    print("Naive Bayes Accuracy: " + str(accuracy))
    
'''
Counting and saving stuff => {0(class):{0(index):{1(count for bin1),2(count for bin2),.....}, 1(index):{..}..}} 
'''   
def attributeCount(classDictionary):
    storeCount = {}
    count = 1
    #print(classDictionary.items())
    for key , value in classDictionary.items():
        #print(value)
        for num in range(len(value)):          
            for subValue in range(len(value[num])):
                #print(subValue)             
                if(count > len(value[num])):
                    count = 1
                if(key not in storeCount):
                    storeCount[key] = {}
                if(count not in storeCount[key]):
                    storeCount[key][count] = {}
                for i in range(1,11):
                    '''
                    storing every bin value with value 1 and Then after the end of this loop calculating there value
                    Laplacian Method
                    '''
                    if (i not in storeCount[key][count]):
                        storeCount[key][count][i] = 1
                if (value[num][subValue] not in storeCount[key][count]):
                    storeCount[key][count][value[num][subValue]] = 1
                else:
                    x = storeCount[key][count][value[num][subValue]]
                    x = x + 1
                    storeCount[key][count][value[num][subValue]] = x
                count +=1         
    print("Bin count" + str(storeCount))
    priorProbabilityCalculation(storeCount)   
       
def priorProbabilityCalculation(storeCount):
    '''
    These loops are used to get the sum of all the values of class
    '''
    for key, value in storeCount.items():
        for secondaryKey, secondaryValue in value.items():
            count = 0
            for finalKey in secondaryValue.items():
                count += finalKey[1]
            if(key not in totalClassValue):
                totalClassValue[key] = count

    print("Class Probability " + str(totalClassValue))
    '''
    To calculate total of training data
    '''
    total = sum(totalClassValue.values())    
    '''
    Calculation of priori probability
    currently it is doing with whole class but it will be done on class training data divided by
    total training data
    '''
    for key, value in totalClassValue.items():
        if(key not in classProbabilityValue):
            x = value / total
            classProbabilityValue[key] = x
                
    print("Prior Probability is " + str(classProbabilityValue))   
    '''     
    Calculation of conditional probability
    In this block using prior and conditional probability we will calculate Posterior Probability
    Calculate particular bin value count of each class divided by total bin counts
    '''
    for key, value in storeCount.items():
        for secondaryKey, secondaryValue in value.items():
            for lastKey in secondaryValue.items():
                for subKey, subValue in totalClassValue.items():
                    if(key not in conditionalProbabilityValue):
                        conditionalProbabilityValue[key] = {}
                    if(secondaryKey not in conditionalProbabilityValue[key]):
                        conditionalProbabilityValue[key][secondaryKey] = {}
                    if (subKey == key): 
                        conditionalProbabilityValue[key][secondaryKey][lastKey[0]] = lastKey[1]/subValue
    print("Conditional probability => " + str(conditionalProbabilityValue))

def testingNaiveBayes(testData): 
    testIndex = testData
    tempPosterior = 0 
    posteriorProbability = classProbabilityValue
    predictionDictionary = {}
    #Before Class Values
    for conditionalKey, conditionalValue in conditionalProbabilityValue.items():
        tempPosterior = 0      
        for subKey, subValue in conditionalValue.items():
            for finalValue in subValue.items():
                if(testIndex[subKey-1] == finalValue[0]):
                    tempPosterior = tempPosterior + math.log(finalValue[1])
                    break
        if(conditionalKey not in predictionDictionary):
            predictionDictionary[conditionalKey] = tempPosterior
    #After Class Values            
    for posteriorKey, posteriorValue in posteriorProbability.items():
        for predictionKey, predictionValue in predictionDictionary.items():
            if (posteriorKey == predictionKey):
                predictionValue = predictionValue + math.log(posteriorValue)
                predictionDictionary[posteriorKey] = predictionValue
    
    return (max(predictionDictionary, key=lambda i:predictionDictionary[i]))
                   
                    
def experiment_NaiveBayes():
    glass = getGlass()
    dataSet = glass[0]
    labels = glass[1]
    trainData = dataSet[::2]
    trainLabels = labels[::2]
    testData = dataSet[1::2]
    testLabels = labels[1::2]
    classSeperation(trainData, trainLabels, testData, testLabels)

def experiment_ID3():
    #data = getBreastCancer()
    #data = getIris()
    #data = getGlass()
    #data = getSoybean()
    data = getVote()
    dataSet = data[0]
    labels = data[1]
    # Contrived experiment, divide test set evenly amongst examples
    trainData = dataSet[::3]
    trainLabels = labels[::3]
    testData = dataSet[1::3]
    testLabels = labels[1::3]
    validationData = dataSet[2::3]
    validationLabels = labels[2::3]
    run_ID3(trainData, trainLabels, testData, testLabels, validationData, validationLabels)    

################################
#KNN algorithm
################################

def divideData(data, labels) : 
    trainData = data[::3]
    trainLabels = labels[::3]
    testData = data[1::3]
    testLabels = labels[1::3]
    print("trainData")
    print(trainLabels)
    print("testData")
    print(testLabels)
    return (trainData, trainLabels, testData, testLabels)

def vdm(x,y) :
    pass
	
def calculateListKneighbors(trainData, trainLabels, currentRaw, k):
    listDist = list()
    for i in range(0, len(trainData)):
        dist = vdm(currentRaw, trainData[i])
        listDist.append(dist,trainLabels[i])
    sortedList = listDist.sort()
    kNeighbors = list()
    for i in range(0, k):
        kNeighbors.append(sortedList[i][1])
    return kNeighbors

def selectClass(kNeighbors):
    dictionaryClass = {}
    for i in range(0, len(kNeighbors)):
        if kNeighbors[i] in dictionaryClass :
            dictionaryClass[kNeighbors[i]] +=1
        else :
            dictionaryClass[kNeighbors[i]] = 1
    return max(dictionaryClass.iteritems(), key=operator.itemgetter(1))[0]

def knn(data, labels, k):
    dataDivided = divideData(data, labels)
    newLabels = list()
    trueVal = 0
	
    #Define labels for the testData
    for i in range(0, len(testData)):
        kNeighbors = calculateListKneighbors(dataDivided[0],dataDivided[1], i, k)
        newClass = selectClass(kNeighbors)
        newLabels.append(newClass)
        if newClass == testLabels[i] :
            trueVal = trueVal + 1
    #Define Accuracy
    accuracy = (trueVal / len(testData)) * 100
    print(accuracy)
		

	
    
def main():
    #tests for KNN
    DATA = getIris()
    newData = divideData(DATA[0],DATA[1])
    #experiment_ID3()
    '''
    #This is the block which i used to call Naive Bayes
    dataValues = getGlass()
    print(dataValues)
    classSeperation(dataValues)
    '''
    
if __name__ == '__main__':
    main()
