import numpy as np

def hypothesis(theta,x):
  return np.dot(x,theta)

def mse_cost(theta,x,y):
  m = len(y)
  h = hypothesis(theta,x)
  return np.dot((h - y).T, h - y)/ m

def gradient_descent(x, y, alpha, n_iteration):
  m = len(y)
  x = np.c_[np.ones(x.shape[0]),x]
  theta = np.zeros((x.shape[1],1))

  for i in range(n_iteration):
    h = hypothesis(theta,x)
    gradient = 2 * np.dot(x.T, h - y ) / m
    temp = theta - alpha * gradient
    theta = temp

  return theta