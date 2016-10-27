'''
CSCI 446 
Fall 2016
Project 3
Group 3

@author: Joe DeBruycker
@author: Shriyansh Kothari
@author: Sara Ounissi
'''

def main():        
    fileOpen = open("D:/Shriyansh_PostGraduation/Artifical Intelligence/Project 3/breast-cancer-wisconsin.data.txt")
    for lines in fileOpen:
        '''
        #Just Checking if reading individual elements in the line selected.
        for element in lines:
            print(element)
        '''
        print(lines.rstrip())
    fileOpen.close()    
    
    
if __name__ == '__main__':
    main()
