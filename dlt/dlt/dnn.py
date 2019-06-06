"""
This file defines neural networks.
"""

from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.losses import binary_crossentropy, categorical_crossentropy
from keras.optimizers import SGD
import numpy as np

class DNNClassifier(Sequential):
    """
    Classifier using fully connected neural network.
    Usage: DNNClassifier([units...])

    activation of hidden layer = relu
    activation of output layer = softmax
    loss = categorical_crossentropy
    optimizer = adam
    """

    def __init__(self, units, optim='sgd'):
        """
        Build model. Based on specified units.
        """
        Sequential.__init__(self)
        self.add(Dense(units[1],input_shape=(units[0],),activation='relu'))
        for i in units[2:-1]:
            self.add(Dense(i,activation='relu'))
        if units[-1] == 1:
            self.add(Dense(units[-1],activation='sigmoid'))
            l = binary_crossentropy
        else:
            self.add(Dense(units[-1],activation='softmax'))
            l = categorical_crossentropy
        self.compile(loss=l, optimizer=optim, metrics=['accuracy'])



class DNNRegressor(Sequential):
	"""
	Regressor using fc nets.
    Usage: DNNRegressor([units...])

	activation of hidden layer = relu
	loss = mse
	optimizer = adam
	"""

	def __init__(self, units, optim='sgd'):
		"""
		Build model. Based on specified units.
		"""
		Sequential.__init__(self)
		self.add(Dense(units[1],input_shape=(units[0],),activation='relu'))
		for i in units[2:-1]:
			 self.add(Dense(i,activation='relu'))
		self.add(Dense(units[-1]))
		self.compile(loss='mse', optimizer=optim)

