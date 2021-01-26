import numpy as np

def hypo(theta,x):
    return np.dot(x,theta)

def sigmoid(z):
    
    s = 1/(1+np.exp(-z))
    return s

def cost(theta,x,y):
    m = len(y)
    z = sigmoid(hypo(theta,x))
    c = - np.dot(y.T,np.log(z)) - np.dot((1-y).T,np.log(1-z))
    return c/m 

def gra(x,y, iteration, alpha):
    m = len(y)
    x = np.c_[np.ones((x.shape[0],1)),x]
    theta = np.random.rand(x.shape[1],1)
        
    for i in range(iteration):
        
        gradient = np.dot(x.T,sigmoid(hypo(theta,x))-y )
        temp = theta - alpha*gradient
        theta = temp
    return theta

def predict_and_score(x,y, theta,threshold):
    x = np.c_[np.ones((x.shape[0],1)),x]
    h = hypo(theta,x)
    s = sigmoid(h)
    y_pred = (s>threshold).astype(int).reshape((len(y),1))
    
    count = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y[i]:
            count+= 1
    score = count/len(y)
    
    return y_pred,score