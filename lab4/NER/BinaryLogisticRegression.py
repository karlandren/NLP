from __future__ import print_function
import math
import random
import numpy as np
import matplotlib.pyplot as plt

"""
This file is part of the computer assignments for the course DD1418/DD2418 Language engineering at KTH.
Created 2017 by Johan Boye, Patrik Jonell and Dmytro Kalpakchi.
"""

class BinaryLogisticRegression(object):
    """
    This class performs binary logistic regression using batch gradient descent
    or stochastic gradient descent
    """

    #  ------------- Hyperparameters ------------------ #

    LEARNING_RATE = 0.1  # The learning rate.
    CONVERGENCE_MARGIN = 0.001  # The convergence criterion.
    MAX_ITERATIONS = 1000 # Maximal number of passes through the datapoints in stochastic gradient descent.
    MINIBATCH_SIZE = 1000 # Minibatch size (only for minibatch gradient descent)
    # ----------------------------------------------------------------------


    def __init__(self, x=None, y=None, theta=None):
        """
        Constructor. Imports the data and labels needed to build theta.

        @param x The input as a DATAPOINT*FEATURES array.
        @param y The labels as a DATAPOINT array.
        @param theta A ready-made model. (instead of x and y)
        """
        if not any([x, y, theta]) or all([x, y, theta]):
            raise Exception('You have to either give x and y or theta')

        if theta:
            self.FEATURES = len(theta)
            self.theta = theta

        elif x and y:
            # Number of datapoints.
            self.DATAPOINTS = len(x)

            # Number of features.
            self.FEATURES = len(x[0]) + 1

            # Encoding of the data points (as a DATAPOINTS x FEATURES size array).
            self.x = np.concatenate((np.ones((self.DATAPOINTS, 1)), np.array(x)), axis=1)

            # Correct labels for the datapoints.
            self.y = np.array(y)

            # The weights we want to learn in the training phase.
            self.theta = np.random.uniform(-1, 1, self.FEATURES)

            # The current gradient.
            self.gradient = np.zeros(self.FEATURES)



    # ----------------------------------------------------------------------


    def sigmoid(self, z):
        """
        The logistic function.
        """
        return 1.0 / ( 1 + math.exp(-z) )


    def conditional_prob(self, label, datapoint):
        """
        Computes the conditional probability P(label|datapoint)
        """

        # REPLACE THE COMMAND BELOW WITH YOUR CODE

        return self.sigmoid(np.dot(self.theta, self.x[datapoint]))


    def compute_gradient_for_all(self):
        """
        Computes the gradient based on the entire dataset
        (used for batch gradient descent).
        """

        # YOUR CODE HERE


    def compute_gradient_minibatch(self, minibatch):
        """
        Computes the gradient based on a minibatch
        (used for minibatch gradient descent).
        """
        
        # YOUR CODE HERE


    def compute_gradient(self, datapoint):
        """
        Computes the gradient based on a single datapoint
        (used for stochastic gradient descent).
        """

        # YOUR CODE HERE


    def stochastic_fit(self):
        """
        Performs Stochastic Gradient Descent.
        """
        # self.init_plot(self.FEATURES)

        self.LEARNING_RATE = 0.1 
        self.CONVERGENCE_MARGIN = 0.0001
        self.MAX_ITERATIONS = 1000

        lamb = 2
        regularizaton = False

        margin = 1
        iterations = 0  
        while margin > self.CONVERGENCE_MARGIN and iterations < self.MAX_ITERATIONS:
            
            i = random.randint(1, self.DATAPOINTS)
    
            for k in range(self.FEATURES):
                self.gradient[k] = self.x[i,k] * self.sigmoid(np.dot(self.theta, self.x[i])) - self.y[i]

            if regularizaton:
                self.gradient += lamb * self.theta

            self.theta -= self.LEARNING_RATE*self.gradient

            # self.update_plot(margin)

            margin = np.sum(np.power(self.gradient, 2))
            iterations += 1
            
        print()
        print("Margin and iterations final: ")
        print(margin)
        print(iterations)
        # YOUR CODE HERE


    def minibatch_fit(self):
        """
        Performs Mini-batch Gradient Descent.
        """
        # self.init_plot(self.FEATURES)

        self.LEARNING_RATE = 0.01 
        self.CONVERGENCE_MARGIN = 0.0005
        self.MAX_ITERATIONS = 10000
        self.MINIBATCH_SIZE = 100

        lamb = 0.001
        regularizaton = True

        margin = 1
        iterations = 0
        while margin > self.CONVERGENCE_MARGIN and iterations < self.MAX_ITERATIONS:
            
            batch_idx = np.random.choice(self.DATAPOINTS, self.MINIBATCH_SIZE)
           
            for k in range(self.FEATURES):
                self.gradient[k] = np.sum(self.x[batch_idx,k]*([self.sigmoid(s) for s in np.dot(self.x[batch_idx], self.theta)]-self.y[batch_idx]))/self.MINIBATCH_SIZE

            if regularizaton:
                self.gradient += lamb * self.theta

            self.theta -= self.LEARNING_RATE*self.gradient
            
            # self.update_plot(np.sum(np.square(self.gradient)))
            print(margin)
            margin = np.sum(np.power(self.gradient, 2))
            iterations += 1

        print()
        print('Margin and iterations final: ')
        print(margin)
        print(iterations)
        # YOUR CODE HERE


    def fit(self):
        """
        Performs Batch Gradient Descent
        """
        self.init_plot(self.FEATURES)

        margin = 1
        while margin > self.CONVERGENCE_MARGIN:
            
            for i in range(self.FEATURES):
                self.gradient[i] = np.sum((self.x[:,i]*([self.sigmoid(s) for s in np.dot(self.x, self.theta)]-self.y)))/self.DATAPOINTS

            self.theta -= self.LEARNING_RATE*self.gradient
            
            self.update_plot(np.sum(np.square(self.gradient)))

            margin = np.sum(np.power(self.gradient, 2))
            print(margin)

        # YOUR CODE HERE


    def classify_datapoints(self, test_data, test_labels):
        """
        Classifies datapoints
        """
        print()
        print('Model parameters:')

        print('  '.join('{:d}: {:.4f}'.format(k, self.theta[k]) for k in range(self.FEATURES)))

        self.DATAPOINTS = len(test_data)

        self.x = np.concatenate((np.ones((self.DATAPOINTS, 1)), np.array(test_data)), axis=1)
        self.y = np.array(test_labels)
        confusion = np.zeros((self.FEATURES, self.FEATURES))

        for d in range(self.DATAPOINTS):
            prob = self.conditional_prob(1, d)
            predicted = 1 if prob > .5 else 0
            confusion[predicted][self.y[d]] += 1

        print('                       Real class')
        print('                 ', end='')
        print(' '.join('{:>8d}'.format(i) for i in range(2)))
        for i in range(2):
            if i == 0:
                print('Predicted class: {:2d} '.format(i), end='')
            else:
                print('                 {:2d} '.format(i), end='')
            print(' '.join('{:>8.3f}'.format(confusion[i][j]) for j in range(2)))

        print()
        print('Accuracy: ', end='')
        print((confusion[0,0]+confusion[1,1])/(confusion[0,0]+confusion[1,1]+confusion[1,0]+confusion[0,1]))
        print('Precision: ', end='')
        print(confusion[1,1]/(confusion[1,1]+confusion[1,0]))
        print('Recall: ', end='')
        print(confusion[1,1]/(confusion[1,1]+confusion[0,1]))


    def print_result(self):
        print(' '.join(['{:.2f}'.format(x) for x in self.theta]))
        print(' '.join(['{:.2f}'.format(x) for x in self.gradient]))


    # ----------------------------------------------------------------------

    def update_plot(self, *args):
        """
        Handles the plotting
        """
        if self.i == []:
            self.i = [0]
        else:
            self.i.append(self.i[-1] + 1)

        for index, val in enumerate(args):
            self.val[index].append(val)
            self.lines[index].set_xdata(self.i)
            self.lines[index].set_ydata(self.val[index])

        self.axes.set_xlim(0, max(self.i) * 1.5)
        self.axes.set_ylim(0, max(max(self.val)) * 1.5)

        plt.draw()
        plt.pause(1e-20)


    def init_plot(self, num_axes):
        """
        num_axes is the number of variables that should be plotted.
        """
        self.i = []
        self.val = []
        plt.ion()
        self.axes = plt.gca()
        self.lines =[]

        for i in range(num_axes):
            self.val.append([])
            self.lines.append([])
            self.lines[i], = self.axes.plot([], self.val[0], '-', c=np.random.rand(3,1), linewidth=1.5, markersize=4)

    # ----------------------------------------------------------------------


def main():
    """
    Tests the code on a toy example.
    """
    x = [
        [ 1,1 ], [ 0,0 ], [ 1,0 ], [ 0,0 ], [ 0,0 ], [ 0,0 ],
        [ 0,0 ], [ 0,0 ], [ 1,1 ], [ 0,0 ], [ 0,0 ], [ 1,0 ],
        [ 1,0 ], [ 0,0 ], [ 1,1 ], [ 0,0 ], [ 1,0 ], [ 0,0 ]
    ]

    #  Encoding of the correct classes for the training material
    y = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0]
    b = BinaryLogisticRegression(x, y)
    b.fit()
    b.print_result()
    b.classify_datapoints(x,y)


if __name__ == '__main__':
    main()
