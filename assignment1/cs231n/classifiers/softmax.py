import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    f_1 = X[i].dot(W)
    f_1 -= np.max(f_1)
    exp = np.exp(f_1)
    p = exp / np.sum(exp)
    loss -= np.log(p[y[i]])
    for j in range(num_classes):
      dW[:,j] += (p[j] - (j==y[i])) * X[i]
  loss /= num_train
  dW /= num_train
  dW += reg*W
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW



def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_classes = W.shape[1]
  num_train = X.shape[0]
  #####################
  f_1 = X.dot(W)
  f_1 = (f_1.T - np.max(f_1,axis=1)).T
  exp = np.exp(f_1)
  p = (exp.T / np.sum(exp, axis=1)).T
  delta0 = np.log([p[i,y[i]] for i in xrange(num_train) ])
  loss = -np.sum(delta0)/num_train

  ind = np.zeros(p.shape)
  ind[range(num_train),y] = 1
  #print(W.shape,X.shape,p.shape)
  dW = np.dot(X.T,(p-ind))

  #####################
  """for i in xrange(num_train):
    f_0 = X[i].dot(W)
    f_0 -= np.max(f_0)
    exp0 = np.exp(f_0)
    p = exp0 / np.sum(exp0)
    delta =  np.log(p[y[i]])
    if i==0:
      print delta
      print(delta0[i])"""
    #for j in range(num_classes):
      #dW[:,j] += (p[j] - (j==y[i])) * X[i]
  #loss /= num_train
  dW /= num_train
  dW += reg*W
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
