#TODO
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import sys


def normalize(lst):
    #print(lst)
    minimum = min(lst)
    maximum = max(lst)
    denomintor = maximum - minimum
    return [ float(elem - minimum)/float(denomintor) for elem in lst ], minimum,  maximum


def initWeight():
    w0 = round(float(random.uniform(0,1)),10)
    w1 = round(float(random.uniform(0,1)),10)
    return (w0,w1)

def validation(records, outcomes):
    FN = 0
    FP = 0
    TN = 0
    TP = 0
    for count in range(len(records)):
        print("outcomes[count]", outcomes[count])
        if(outcomes[count] == 1 and records[count][-1] == 0):
            FP = FP + 1
        if(outcomes[count] == 0 and records[count][-1] == 1):
            FN = FN + 1
        if(outcomes[count] == 0 and records[count][-1] == 0):
            TN = TN + 1
        if(outcomes[count] == 1 and records[count][-1] == 1):
            TP = TP + 1
    
    
    return ( TP, FP, FN, TN)

def accuracy(records, outcomes):
    value = 0
    for count in range(len(records)):
        if records[count][-1] == outcomes[count]:
            value += 1

def setTrainValidateTestRecords(records):
    line_count= len(records)
    testValSize = .10*line_count

    #print("testValSize", int(testValSize))

    testData = []
    validateData = []
    trainData = []

    for count in range(line_count):
        #row = 0
        if(count < testValSize):
            testData = testData + [records[count]]
            #row = row + 1
        elif(count < 2*testValSize):
            validateData = validateData + [records[count]]
        else:
            trainData = trainData + [records[count]]
    return testData,validateData,trainData


url = "kc_house_data.csv"

df = pd.read_csv(url,   nrows=1)
columns = list(df.head(0))
print(columns)
df = pd.read_csv(url, skiprows=1, names=columns)
#print("len(columns)",len(columns))

features = columns[2:6]
#print("len(features)",len(features))
data = df.loc[:, features].values

random.seed(23)
random.shuffle(data)


testData,validateData,trainData = setTrainValidateTestRecords(data)

y = [ d[0] for d in trainData ] 
x  = [ d[3] for d in trainData ]

yvalidateData = [ d[0] for d in validateData ] 
xvalidateData  = [ d[3] for d in validateData ]

ytestData = [ d[0] for d in testData ] 
xtestData  = [ d[3] for d in testData ]



#[ print(d[3]) for d in data ] 

x,minX, maxX = normalize(x)
y,minY, maxY = normalize(y)

size =len(x)
#print(y)

print("X",minX, maxX)
print("Y",minY, maxY)


(w0,w1) = initWeight()
theta=[0,0]

#print(x)
#print(y)


def cost(X, Y, theta):
    J=np.dot((np.dot(X,theta) - Y).T,(np.dot(X,theta) - Y))/(2*len(Y))
    return J
alpha = 0.1 # Specify the learning rate
theta =  np.array([[0,0]]).T # Initial values of theta
X = np.c_[np.ones(size),x]
Y = np.c_[y]
X_1=np.c_[x].T
#num_iters = 10000
num_iters = 1000
cost_history=[]
theta_history=[]
#for i in range(num_iters):

prev = 0
next = sys.float_info.max

delta = .0001
converged = False
count = 0
z = 0
while(not converged):
    a=np.sum(theta[0]- alpha * (1/len(Y)) * np.sum((np.dot(X,theta)- Y)))
    b=np.sum(theta[1] - alpha * (1/len(Y)) * np.sum(np.dot(X_1,(np.dot(X,theta)-Y))))
    
    theta= np.array([[a],[b]])
    cost_history.append(cost(X,Y,theta))
    theta_history.append(theta)
    if(len(theta_history) > 2 ):
        if(theta_history[count][1]- theta_history[count-1][1] <delta):
            converged = True

    count+= 1
    
convert = (maxY-minY) + minY
print(theta)
#print(x[0:10])
#print(y[0:10])
res = [ (theta[1]*elem + theta[0])*convert for elem in x ]
print("Projection from training data of 1st 10 elements", res[0:10])
#[[ 2.51030574]
# [ 0.95243058]]

x,minX, maxX = normalize(xtestData)
y,minY, maxY = normalize(ytestData)
convert = (maxY-minY) + minY
res = [ (theta[1]*elem + theta[0])*convert for elem in x ]
print("/******************************/")
print("Projection from test data", res[0:10])
print("Actual data from test price", ytestData[0:10])



x,minX, maxX = normalize(xvalidateData)
y,minY, maxY = normalize(yvalidateData)
convert = (maxY-minY) + minY
res = [ (theta[1]*elem + theta[0])*convert for elem in x ]
print("/******************************/")
print("Projection from validation data", res[0:10])

print("Actual data from validation price", ytestData[0:10])


'''
legend = ['sqft_living', 'Price', 'Bathrooms']
sqft_living = [ d[3] for d in data if d[3] != 0] 
Price = [ d[0] for d in data if d[0] != 0] 
Bathrooms = [ d[2] for d in data if d[2] != 0] 


plt.hist([sqft_living, Price, Bathrooms], color=['orange', 'green', 'blue'])
plt.xlabel("Feature")
plt.ylabel("Frequency")
plt.legend(legend)
plt.xticks(range(0, .5))
plt.yticks(range(1, 10))
plt.title('Predict Housing Prices - Simple Linear Regression')
plt.show()
'''