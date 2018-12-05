import pandas as pd
import numpy as np

def gradient_descent (X,Y,thetas):
    
    iterations = 10000
    learning_factor=0.002
    n = Y.size
    
    i=0
    while (i<iterations):
        
        #find predicted value of Y
        y_predicted = np.dot(X,thetas)
        
        #find hTheta as logistic function of y_predicted
        hTheta = 1/(1+np.exp(-y_predicted))
        
        #calculate cost/loss function
        cost = -1/n*(sum(Y*np.log(hTheta)+(1-Y)*(np.log(1-hTheta))))
        
        #calculate gradient on partial differential of cost for all thetas
        dTheta = 1/n*(sum((hTheta-Y)*X))
        dTheta = dTheta.reshape(4,1)
        
        #apply gradient descent and recalculate thetas
        thetas = thetas - learning_factor*dTheta
        
        i=i+1
        
    print ('cost is {} thetas are {}'.format(cost, thetas.T))
    
    return thetas

#initialize vector X which has 3 features
x1=(1,2,3,4,5,6,7,8,9,10)
x2=(11,22,33,44,55,66,77,88,99,100)
x3=(5,6,9,12,45,77,88,99,11,12)
xlist = list (zip(x1,x2,x2))
df = pd.DataFrame(xlist)
X = df.values

#add a ones column for the intercept co-effecient and add that column to X
ones = np.ones((10,1),dtype=int)
X=np.concatenate((ones,X),axis=1)

#initialize vector Y where y=(0,1) for different values of X vector
y1=(0,0,0,0,1,1,1,1,1,0)
Y=np.array(y1)
Y=Y.reshape((10,1))

#initialize thetas vector which would hold the co-effecients 
thetas = np.zeros((4,1),dtype=int)

thetas = gradient_descent(X,Y,thetas)

#predict the probality of Y=1 for given X parameterixed by Theta
testX=np.array([1,8,88,99])
testX=testX.reshape(1,4)
probality=1/(1+np.exp(np.dot(testX,thetas)))
predicted = 0
if (probality>=0.5):
    predicted = 1
print ('Probablity of Y being 1 for given test X and parameterised by Theta is {} and its class is {}'.format(probality,predicted))