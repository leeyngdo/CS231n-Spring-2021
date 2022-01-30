from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        W_first = np.random.randn(input_dim, hidden_dim) * weight_scale
        W_second = np.random.randn(hidden_dim, num_classes) * weight_scale
        b_first = 0.0
        b_second = 0.0
        
        self.params['W1'] = W_first
        self.params['W2'] = W_second
        self.params['b1'] = b_first
        self.params['b2'] = b_second

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']
        
        input_layer = np.dot(X, W1) + b1
        hidden_layer = np.maximum(0, input_layer)
        output_layer = np.dot(hidden_layer, W2) + b2
        scores = output_layer

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        num_train, num_classes = scores.shape
        scores_matrix = np.copy(scores)
        
        ####### loss
        scores_matrix -= np.max(scores_matrix[np.arange(num_train)])
        softmax_matrix = np.exp(scores_matrix)/(np.sum(np.exp(scores_matrix), axis = 1).reshape(-1, 1))

        loss_matrix = -np.log(softmax_matrix[np.arange(num_train), y])
        loss = np.sum(loss_matrix)

        # Average and L2 regularization
        loss /= num_train
        loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))

        ####### grad (backpropagation)
        dscores = np.copy(softmax_matrix) # (N, C) 
        dscores[np.arange(num_train), y] += -1
        dscores /= num_train
        
        grads['W2'] = np.dot(hidden_layer.T, dscores) + self.reg * W2 # (H, N) X (N, C)
        
        broadcast = np.full((num_train, 1), 1) # h * W2 + b2 -> b2 will be broadcasted by broadcast.dot(b2)
        grads['b2'] = np.dot(broadcast.T, dscores)
        
        dinput = np.dot(softmax_matrix, W2.T) # (N, C) X (C, H) 
        coeff_matrix = np.zeros((num_train, num_classes))
        coeff_matrix[np.arange(num_train), y] += -1
        dinput += np.dot(coeff_matrix, W2.T) 
        dinput = dinput * (input_layer > 0) / num_train 
        
        broadcast = np.full((num_train, 1), 1) # x * W1 + b1 -> b1 will be broadcasted by broadcast.dot(b2)
        grads['b1'] = np.dot(broadcast.T, dinput)
        grads['W1'] = np.dot(X.T, dinput) + self.reg * W1 # (D, N) X (N, H)  

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
