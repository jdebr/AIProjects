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
    ''' Reads in the file specified and parses the content into numerical
    data as a list of lists
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
        
        

def main():
    #readingFiles()
    fileName = "../data/iris.data"
    irisData = readFile(fileName)
    print(irisData)
    
if __name__ == '__main__':
    main()
