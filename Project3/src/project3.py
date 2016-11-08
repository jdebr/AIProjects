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

def main():
    irisData = getSoybean()
    print(irisData[0])
    print(irisData[1])
    
if __name__ == '__main__':
    main()
