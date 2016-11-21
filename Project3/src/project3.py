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
import ID3Node
import TanNode
import operator
from statistics import mean 

conditionalProbabilityValue = {}
totalClassValue = {}
classProbabilityValue = {}            
            
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
            
def getIris():
    ''' Processes the Iris data, returning a tuple of 2 lists: ([data],[labels]),
    where the data has been turned into lists of numerical discretized data and 
    class labels are strings 
    '''
    iris = readFile("../data/iris.data")
    data = list()
    labels = list()
    # Iris has 5 attributes, the last being the class
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
            # Encode data as 'y' -> 1, 'n'-> -1, '?'-> 0
            if item[i] == 'y':
                newData.append(1)
            elif item[i] == 'n':
                newData.append(-1)
            else:
                newData.append(0)
        data.append(newData)
        labels.append(item[0])
    
    return (data, labels)

'''
TAN
'''
def calculate_CMI(feat1, feat2, dataSet, labels):
    ''' Calculates the Conditional Mutual Information between two random
    variables: feat1 and feat2, which are given as integer indexes of the 
    individual data points, represented as lists in the dataSet list
    '''
    # Get class distribution from labels
    classCounts = {}
    for label in labels:
        if label in classCounts:
            classCounts[label] += 1
        else:
            classCounts[label] = 1
            
    # Get feat1 distribution
    feat1Counts = {}
    for data in dataSet:
        if data[feat1] in feat1Counts:
            feat1Counts[data[feat1]] += 1
        else:
            feat1Counts[data[feat1]] = 1
            
    # Get feat1 distribution
    feat2Counts = {}
    for data in dataSet:
        if data[feat2] in feat2Counts:
            feat2Counts[data[feat2]] += 1
        else:
            feat2Counts[data[feat2]] = 1
    
    # CMI calculation
    CMI = 0.0
    # Sum over all classes
    for c in classCounts.keys():
        # Sum over values of feat1
        for x in feat1Counts.keys():
            # Sum over values of feat2
            for y in feat2Counts.keys():
                # Get counts for conditional probabilities of values
                localCount = 0.0
                xCount = 0.0
                yCount = 0.0
                for i in range(len(dataSet)):
                    if labels[i] == c and dataSet[i][feat1] == x and dataSet[i][feat2] == y:
                        localCount += 1
                    if labels[i] == c and dataSet[i][feat1] == x:
                        xCount += 1
                    if labels[i] == c and dataSet[i][feat2] == y:
                        yCount += 1
                # Prob of (x,y,c)
                Pxyc = localCount/len(dataSet)
                ##print("Prob of C={0}, X={1}, Y={2}: {3}".format(c, x, y, Pcxy))
                # Only proceed for nonzero probs
                if Pxyc > 0:
                    # Prob of (x,y | c)
                    Pxy = localCount/classCounts[c]
                    # Probs of (x|c) and (y|c)
                    Px = xCount/classCounts[c]
                    Py = yCount/classCounts[c]
                    # Add to CMI
                    CMI += (Pxyc * math.log2(Pxy/(Px*Py)))
    
    ##print("CMI: " + str(CMI))
    return CMI
    
def run_TAN(dataSet, labels, testData, testLabels):
    ''' Build a complete undirected graph where each node represents an 
    attribute in the dataset and each edge is weighted by the Conditional
    Mutual Information value between each pair of features
    '''
    # Network is a list of nodes that represent the features 
    network = list()
    # Edges is a list of edge IDs and weights for maximum spanning tree
    edges = list()
    # Forest is a list of sets of trees for the max spanning alg
    forest = list()
    # Build a node for each feature in the dataset, ID'd by index
    #print("####################")
    #print("Building Complete Undirected Graph From Features")
    for i in range(len(dataSet[0])):
        node = TanNode.TanNode(str(i))
        network.append(node)
        # Prep forest
        forest.append({str(i)})
    # Connect all nodes and set the weights for each edge as the CMI between those features
    for i in range(len(dataSet[0])-1):
        for j in range(i+1, len(dataSet[0])):
            #network[i].addUndirectedEdge(network[j])
            weight = calculate_CMI(i, j, dataSet, labels)
            #print("CMI between feature {0} and feature {1}: {2}".format(i,j,weight))
            #network[i].setWeight(j, weight)
            #network[j].setWeight(i, weight)
            edges.append((str(i), str(j), weight))
    # Build maximum weight spanning tree
    #print("####################")
    #print("Building Max Weight Spanning Tree")
    edges = sorted(edges, key = lambda x: x[2], reverse = True)
    spanTree = list()
    while edges and len(spanTree) < len(network)-1:
        maxEdge = edges.pop(0)
        isCycle = False
        tree1 = None
        tree2 = None
        # Look for cycles
        for index, tree in enumerate(forest):
            if maxEdge[0] in tree and maxEdge[1] in tree:
                isCycle = True
                break
            elif maxEdge[0] in tree:
                tree1 = index
            elif maxEdge[1] in tree:
                tree2 = index
        # If no cycles, add edge to max spanning tree
        if not isCycle:
            spanTree.append(maxEdge)
            forest[tree1] = forest[tree1].union(forest[tree2])
            del(forest[tree2])
            
    #print("Max Spanning Tree: " + str(spanTree))
    #print("####################")
    # Pick root
    root = random.choice(network)
    ##print("Root: " + root.name)
    # Set directed edges away from root
    edges = spanTree.copy()
    remainingNodes = []
    for edge in edges:
        if edge[0] == root.name:
            root.addDirectedEdge(network[int(edge[1])])
            remainingNodes.append(int(edge[1]))
            spanTree.remove(edge)
        elif edge[1] == root.name:
            root.addDirectedEdge(network[int(edge[0])])
            remainingNodes.append(int(edge[0]))
            spanTree.remove(edge)
    # Set rest of directed edges
    while remainingNodes:
        i = remainingNodes.pop(0)
        currentNode = network[i]
        edges = spanTree.copy()
        for edge in edges:
            if edge[0] == currentNode.name:
                currentNode.addDirectedEdge(network[int(edge[1])])
                remainingNodes.append(int(edge[1]))
                spanTree.remove(edge)
            elif edge[1] == currentNode.name:
                currentNode.addDirectedEdge(network[int(edge[0])])
                remainingNodes.append(int(edge[0]))
                spanTree.remove(edge)
    
    # Use tree to make predictions
    numCorrect = 0
    for i in range(len(testData)):
        testD = testData[i]
        testL = testLabels[i]
        # Initialize classes and class counts
        predictions = {}
        classCounts = {}
        for label in labels:
            if label not in predictions:
                predictions[label] = 0
                classCounts[label] = 1
            else:
                classCounts[label] += 1
                
        # Calculation Probabilities
        for c in predictions.keys():
            # Prob of class
            pC = (float(classCounts[c])/float(len(dataSet)))
            # Prob of root given class
            ##print("Root: " + root.name)
            r = int(root.name)
            localCount = 0 # SMOOTHING?
            for index, data in enumerate(dataSet):
                if labels[index] == c and data[r] == testD[r]:
                    localCount += 1
            pR = (float(localCount)/float(classCounts[c]))
            finalProb = pC * pR
            # Prob of other features
            givenIndex = r
            remainingFeatures = list()
            # Start with children of root
            for child in root.children:
                remainingFeatures.append(int(child.name))
            # Calculate probability of remaining features
            while remainingFeatures:
                localCount = 0.1
                localTotal = 1
                currentFeature = remainingFeatures.pop(0)
                # If current node has children, add them to list
                for child in network[currentFeature].children:
                    remainingFeatures.append(int(child.name))
                # If current node has different parent, change given Value and Index
                if network[currentFeature].parent != givenIndex:
                    givenIndex = network[currentFeature].parent
                for index, data in enumerate(dataSet):
                    if labels[index] == c and data[givenIndex] == testD[givenIndex]:
                        localTotal += 1
                        if data[currentFeature] == testD[currentFeature]:
                            localCount += 1
                localProb = (float(localCount)/float(localTotal))
                # If 
                ##print("Local Prob: " + str(localProb))
                finalProb *= localProb
            # Store probability of each class
            predictions[c] = finalProb
        # Choose prediction as highest probability class
        #print("Calculated Posterior Probabilities: " + str(predictions))
        prediction = max(predictions, key = lambda i:predictions[i])
        #print("Prediction: " + (prediction))
        #print("Actual: " + testL)
        if testL == prediction:
            numCorrect += 1
    #print("####################")        
    #print("Accuracy: " + str(float(numCorrect)/len(testData)))
           
            
    
    
    #for node in network:
    #    #print("Node {0}:".format(node.name))
    #    #print("  connections: " + str(node.childNames))
        
    
def experiment_TAN():
    #print("Running TAN on Vote Data")
    data = getVote()
    dataSet = data[0]
    labels = data[1]
    getResults = crossValidation(dataSet,labels)
    run_TAN(getResults[0], getResults[1], getResults[2], getResults[3])
    
'''
ID3
'''
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
    ##print(classCounts)
    
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
    ##print("Num Splits: " + str(len(featValues)))
        
    # For each value feature can take on measure entropy of that subset
    for value in featValues:
        subData = list()
        subLabels = list()
        # Get subset of data for that value
        for i in range(len(dataSet)):
            if dataSet[i][feature] == value:
                subData.append(dataSet[i])
                subLabels.append(labels[i])
        ##print("Subdata: " + str(subData))
        ##print("Sublabels: " + str(subLabels))
        
        # Get subEntropy, multiply by probability, and add to newEntropy
        subEntropy = calculateEntropy(subData, subLabels)
        p = float(len(subData))/float(len(dataSet))
        expectedEntropy += (subEntropy * p)
        intrinsicValue += (p * math.log2(p))
        ##print("Subentropy: " + str(subEntropy))
        
        
    # Calculate gain
    gain = entropy - expectedEntropy
    ##print("Gain: " + str(gain))
    ##print("Expected Entropy: " + str(expectedEntropy))
    ##print("Intrinsic Value: " + str(-intrinsicValue))
    
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
    #print("##################################")
    #print("Attempting to prune with reduced error pruning technique")
    prune(ID3, ID3, validationData, validationLabels)
    
    # Test decision tree using testData, return classification accuracy
    #print("##################################")
    #print("Running classification with test dataset")
    total = float(len(testData))
    numCorrect = 0
    for i in range(len(testData)):
        prediction = ID3.test(testData[i])
        if testLabels[i] == prediction:
            numCorrect += 1
        #print("ID3 Prediction: " + str(prediction))
        #print("Actual Class: " + testLabels[i])
            
    accuracy = numCorrect/total
    #print("Number correctly classified: " + str(numCorrect))
    #print("ID3 Accuracy: " + str(accuracy))

def build_ID3(trainData, trainLabels, features):
    ''' Build a node of the decision tree for ID3 by choosing the attribute from features that
    results in the highest gain ratio when splitting the trainData and creating child nodes for 
    all possible values of that attribute.  If features is empty, return the majority label of the 
    trainLabels.  If all trainLabels are the same, return that label.
    '''
    #print("################")
    #print("Building Decision Tree")
    root = ID3Node.ID3Node()
    
    # If all trainLabels are the same return the node with that label
    nodeLabel = trainLabels[0]
    allSame = True
    for label in trainLabels:
        if label != nodeLabel:
            allSame = False
            break
    if allSame:
        #print("Leaf Node with class label " + (str(nodeLabel)))
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
        #print("Leaf Node with class label " + (str(majorityLabel)))
        return root
    
    # Otherwise find feature with highest gain ratio and split dataset by creating children nodes
    bestFeature = features[0]
    bestGR = calculateGainRatio(bestFeature, trainData, trainLabels)
    for feature in features:
        currentGR = calculateGainRatio(feature, trainData, trainLabels)
        #print("Calculating Gain Ratio for feature " + str(feature) + ": " + str(currentGR))
        if currentGR > bestGR:
            bestFeature = feature
            bestGR = currentGR
    root.setFeature(bestFeature)
    #print("Best Feature is feature # " + str(bestFeature))
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
    #print("Pruned accuracy: " + str(new_accuracy))
    # Set label back to none if this results in decreased accuracy
    if new_accuracy < accuracy:
        node.label = None
    # Otherwise keep the pruned tree and set the best accuracy
    #else:
        #print("Accuracy is as good or better! Node pruned!")
    return

def experiment_ID3():
    ''' Method for testing ID3 operation '''
    data = getBreastCancer()
    #data = getIris()
    #data = getGlass()
    #data = getSoybean()
    #data = getVote()
    dataSet = data[0]
    labels = data[1]
    # Contrived experiment, divide test set evenly amongst examples, change to randomized CV for final experiment!
    trainData = dataSet[::3]
    trainLabels = labels[::3]
    testData = dataSet[1::3]
    testLabels = labels[1::3]
    validationData = dataSet[2::3]
    validationLabels = labels[2::3]
    #print("Running ID3 on Breast Cancer Data")
    run_ID3(trainData, trainLabels, testData, testLabels, validationData, validationLabels)    


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

    #print("Class Dictionary after joining training labels and data: ")
    #print(classDictionary)
    #print("--------------")
    attributeCount(classDictionary)
    
    # Test Naive Bayes using testData
    totalTest = float(len(testData))
    #print("Size of test set " + str(totalTest))
    numCorrect = 0
    for i in range(len(testData)):
        prediction = testingNaiveBayes(testData[i])
        if testLabels[i] == prediction:
            numCorrect += 1
            
    accuracy = numCorrect/totalTest
    #print("Number correctly classified: " + str(numCorrect))
    #print("Naive Bayes Accuracy: " + str(accuracy))
    
'''
Counting and saving stuff => {0(class):{0(index):{1(count for bin1),2(count for bin2),.....}, 1(index):{..}..}} 
'''   
def attributeCount(classDictionary):
    storeCount = {}
    count = 1
    ##print(classDictionary.items())
    for key , value in classDictionary.items():
        ##print(value)
        for num in range(len(value)):          
            for subValue in range(len(value[num])):
                ##print(subValue)             
                if(count > len(value[num])):
                    count = 1
                if(key not in storeCount):
                    storeCount[key] = {}
                if(count not in storeCount[key]):
                    storeCount[key][count] = {}
                for i in range(1,11):
                    '''
                    storing every bin value with value 1 and Then after the end of this loop calculating there value
                    Additive Smoothing (Laplace) Method
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
    #print("Total number of feature values ({Class: {Feature Index: {Value: Count}}})" + str(storeCount))
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

    #print("Class Counts " + str(totalClassValue))
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
                
    #print("Prior Probability is " + str(classProbabilityValue))   
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
    #print("Conditional probability => " + str(conditionalProbabilityValue))

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
    #print("Running Naive Bayes on Iris Data")
    glass = getIris()
    dataSet = glass[0]
    labels = glass[1]
    for testRuns in range(1):
        getResults = crossValidation(dataSet,labels)
        classSeperation(getResults[0], getResults[1], getResults[2], getResults[3])


################################
#KNN algorithm
################################

#DEviding the data between a training set and a test set 
def divideData(data, labels) : 
    trainData = data[::3]
    trainLabels = labels[::3]
    testData = data[1::3]
    testLabels = labels[1::3]
    ##print("trainData")
    ##print(trainLabels)
    ##print("testData")
    ##print(testLabels)
    return (trainData, trainLabels, testData, testLabels)

#Function that return a dictionary of dictionary
#For each class as a key we have a dictionary of the value/feature and their occurence in the class
def creaListOccInClass(trainData, trainLabels):
    '''
    occInClass = {}
    for i in range(0, len(labels)):
        if labels[i] in occInClass :
            for j in range(0,len(data[0])):
                if data[i][j] in occInClass[labels[i]]:
                    occInClass[labels[i]][data[i][j]] += 1
                else :
                    occInClass[labels[i]][data[i][j]] = 1 
        else :
            occInClass[labels[i]] = {}
            for j in range(0,len(data[0])):
                if data[i][j] in occInClass[labels[i]]:
                    occInClass[labels[i]][data[i][j]] += 1
                else :
                    occInClass[labels[i]][data[i][j]] = 1
    return occInClass
    '''
    classDictionary = {}
    for value in range(len(trainData)):
        if (trainLabels[value] not in classDictionary):
            classDictionary[trainLabels[value]] = [] 
        classDictionary[trainLabels[value]].append(trainData[value])

    ##print("Class Dictionary after joining training labels and data")
    ##print(classDictionary)
    #attributeCount(classDictionary)
    storeCount = {}
    count = 1
    ##print(classDictionary.items())
    for key , value in classDictionary.items():
        ##print(value)
        for num in range(len(value)):          
            for subValue in range(len(value[num])):
                ##print(subValue)             
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
    ##print("Bin count" + str(storeCount))
    return storeCount

#Function that creat a dictionary with each value/feature and it's occurence in the set of data 
def creatOccTot(trainData, trainLabels):
    classDictionary = {}
    for value in range(len(trainData)):
        if (trainLabels[value] not in classDictionary):
            classDictionary[trainLabels[value]] = [] 
        classDictionary[trainLabels[value]].append(trainData[value])

    ##print("Class Dictionary after joining training labels and data")
    ##print(classDictionary)
    #attributeCount(classDictionary)
    storeCount = {}
    count = 1
    ##print(classDictionary.items())
    for key , value in classDictionary.items():
        
        ##print(value)
        for num in range(len(value)):          
            for subValue in range(len(value[num])):
                ##print(subValue)             
                if(count > len(value[num])):
                    count = 1
                if(count not in storeCount):
                    storeCount[count] = {}
                #if(count not in storeCount[key]):
                #    storeCount[key][count] = {}
                for i in range(1,11):
                    '''
                    storing every bin value with value 1 and Then after the end of this loop calculating there value
                    Laplace Method
                    '''
                    if (i not in storeCount[count]):
                        storeCount[count][i] = 1
                if (value[num][subValue] not in storeCount[count]):
                    storeCount[count][value[num][subValue]] = 1
                else:
                    x = storeCount[count][value[num][subValue]]
                    x = x + 1
                    storeCount[count][value[num][subValue]] = x
                count +=1         
    ##print(str(storeCount))
    return storeCount

#Function that return a list containing one occurence of the different classes
def creatListClass(labels):
    listClass = list()
    for i in range(0, len(labels)):
        if labels[i] not in listClass : 
            listClass.append(labels[i])
    return listClass

#vdm computation 
def vdm(x,y,listClass,occInClass,occTot) :
    vdm = 0

    for i in range(1, len(listClass)+1):
        ##print(listClass[i-1])
        ##print(occInClass[listClass[i-1]][i][x])
        ##print(occTot[i][x])
        if y in occInClass[listClass[i-1]][i] and y in occTot[i] and x in occInClass[listClass[i-1]][i] and x in occTot[i]  : 
            ##print(occInClass[listClass[i-1]][i][y])
            ##print(occTot[i][y])
            vdm += abs((occInClass[listClass[i-1]][i][x]/occTot[i][x])-(occInClass[listClass[i-1]][i][y]/occTot[i][y]))
        if (y not in occInClass[listClass[i-1]][i] or y not in occTot[i]) and (x in occInClass[listClass[i-1]][i] and x in occTot[i]):
            vdm += abs((occInClass[listClass[i-1]][i][x]/occTot[i][x]))
        if (x not in occInClass[listClass[i-1]][i] and x not in occTot[i]) and (y in occInClass[listClass[i-1]][i] and y in occTot[i]) :
            vdm += abs(-(occInClass[listClass[i-1]][i][y]/occTot[i][y]))
    return vdm 

#distance function where we sum the VDM of each features 
def distFunctionVDM(x,y,listClass, occInClass, occTot):
    sumVDM = 0
    for k in range(0, len(x)):
        sumVDM += vdm(x[k],y[k],listClass, occInClass, occTot)
    return sumVDM

#Return the list of the kNeighbors, it's a list cotaining only the labels of the kNeighbors
def calculateListKneighbors(trainData, trainLabels, currentRaw, k):
    occInClass = creaListOccInClass(trainData, trainLabels)
    occTot = creatOccTot(trainData, trainLabels)
    listClass = creatListClass(trainLabels)
    listDist = list()
    for i in range(0, len(trainData)):
        dist = distFunctionVDM(trainData[i],currentRaw,listClass, occInClass, occTot)
        listDist.append((dist,trainLabels[i]))
    sortedList = sorted(listDist, key=lambda x:x[0])
    kNeighbors = list()
    for i in range(0, k):
        kNeighbors.append(sortedList[i][1])
    return kNeighbors

#Function that select the predicted class, it looks at all the classes of the 
#kNeighbors and return the one that have the highest occurence 
def selectClass(kNeighbors):
    dictionaryClass = {}
    for i in range(0, len(kNeighbors)):
        if kNeighbors[i] in dictionaryClass :
            dictionaryClass[kNeighbors[i]] +=1
        else :
            dictionaryClass[kNeighbors[i]] = 1
    return max(dictionaryClass, key = lambda x:dictionaryClass[x])

#KNN algorithm
def knn(dataDivided, k):
    #print("KNN Algorithm ................................")
    #print("With k = "+str(k))
    #dataDivided = divideData(data, labels)
    #New list of labels predicted
    newLabels = list()
    trueVal = 0
    #Define labels for the testData
    for i in range(0, len(dataDivided[2])):
        #Creat the list of kNeighbors of each raw of the DataTest
        #print("DataTest["+str(i)+"] :")
        #print(dataDivided[2][i])
        kNeighbors = calculateListKneighbors(dataDivided[0],dataDivided[1], dataDivided[2][i], k)
        #print("kNeighbors for raw number "+str(i))
        #print(kNeighbors)
        #Select the new class of the raw
        newClass = selectClass(kNeighbors)
        #print("Class prediction for the current raw")
        #print(newClass)
        newLabels.append(newClass)
        if newClass == dataDivided[3][i] :
            trueVal = trueVal + 1
    #Define Accuracy
    accuracy = (trueVal / len(dataDivided[2])) * 100
    #print("Accuracy "+str(accuracy))
    return accuracy 

#5x2 Validation
def crossValidation(dataSet,labels, validationSize = 0.1):
    ##print(len(labels))
    tempDictionary = {}
    for value in range(len(dataSet)):
        if (labels[value] not in tempDictionary):
            tempDictionary[labels[value]] = [] 
        tempDictionary[labels[value]].append(dataSet[value])
    ##print(tempDictionary)
    tempTestData = []
    testLabels = []
    tempTrainData = []
    trainLabels = []
    validationData = []
    validationLabels = []
    for key, value in tempDictionary.items():
        random.shuffle(value)
        half = int(math.ceil(len(value)/2))
        tempTestData.append(value[:half])
        tempTrainData.append(value[half:])
        for num in range(0,half):
            testLabels.append(key)
            trainLabels.append(key)
    
    testData = []
    trainData = []
    for i in range(len(tempTrainData)):
        for k in tempTrainData[i]:
            trainData.append(k)
    for i in range(len(tempTestData)):
        for k in tempTestData[i]:
            testData.append(k)
            
#     #print("Equal?")
#     eq = tempTrainData[0][0] == trainData[0]
#     #print(eq)
            
#     #print("--------------")
#     #print("Train Data: " + str(trainData))
#     #print("Train Labels: " + str(trainLabels))
#     #print("Test Data: " + str(testData))
#     #print("Test Labels: " + str(testLabels))
#     #print("--------------")
    
    return(trainData, trainLabels, testData, testLabels, validationData, validationLabels)

def finalExperiment(data, k):
    results = {"knn":[],
               "nb":[],
               "tan":[],
               "id3":[] }
    # 5 passes of 2-fold cross validation for each dataset for each algorithm
    for i in range(5):
        dataSplits = crossValidation(data[0], data[1])
        trainData = dataSplits[0]
        trainLabels = dataSplits[1]
        testData = dataSplits[2]
        testLabels = dataSplits[3]
        validationData = dataSplits[4]
        validationLabels = dataSplits[5]
        
        # KNN
        acc1 = knn((trainData,trainLabels,testData,testLabels), k)
        acc2 = knn((testData,testLabels,trainData,trainLabels), k)
        results["knn"].append(acc1)
        results["knn"].append(acc2)
        
        # NB
        
        # TAN
        
        # ID3
    print(mean(results["knn"]))
    

def main():
    data = getGlass()
    #data = getBreastCancer()
    #data = getSoybean()
    #data = getVote()
    #data = getIris()
    for k in [1,9]:
        finalExperiment(data,k)
    #data = getSoybean()
    # SAMPLE RUN #printOUTS
    #knn(data[0],data[1], 3)
    #experiment_NaiveBayes()
    #experiment_ID3()
    #experiment_TAN()
    #crossValidation(data[0], data[1])
    
if __name__ == '__main__':
    main()
