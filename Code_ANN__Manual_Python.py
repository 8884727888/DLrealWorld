#### a simple ANN by python 
import numpy as np
import pandas as pd 

### creating sigmoid manually, 1/1+e pwer -x
def sigmoid(x,dvt=False):
    if(dvt==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
    
# input  as array 
X = np.array([  [0,0,1,0],
                [0,1,1,1],
                [1,0,1,0],
                [1,1,1,0] ])
    
# output array , transposed            
y = np.array([[0,1,0,1]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
##np.random.seed(1)

# initialize weights randomly with mean 0
Weight = 2*np.random.random((4,1)) - 1

for i in range(500):

    # forward propagation
    input = X
    SigO = sigmoid(np.dot(input,Weight))
    print("input...")
    print(input)
    print("initial weight...")
    print(Weight)
    print("dot product(input*weight) >> sigmoid")
    print(SigO)
    # how much we missed...
    SigO_error = y - SigO
    print("the residual...")
    print(SigO_error)
    # multiply how much we missed by the 
    # slope of the sigmoid .....
    SigO_delta = SigO_error * sigmoid(SigO,True)
    print("how much missed by slope at SigO")
    print(SigO_delta)
    # update weights
    Weight += np.dot(input.T,SigO_delta)
    print("updating weight for next iteration ..")
    print(Weight)

print ("Output After Training:")
print (SigO)
