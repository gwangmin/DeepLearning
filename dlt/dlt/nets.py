"""
This file defines neural networks.
"""

from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import Adam
import numpy as np	

class DNNClassifier(Sequential):
	"""
	Classifier using fully connected neural network.

	activation of hidden layer = relu
	activation of output layer = softmax
	loss = categorical_crossentropy
	optimizer = adam
	"""

	def __init__(self, units):
		"""
		Build model. Based on specified units.
		"""
		Sequential.__init__(self)
		self.add(Dense(units[1],input_shape=(units[0],),activation='relu'))
		for i in units[2:-1]:
			 self.add(Dense(i,activation='relu'))
		self.add(Dense(units[-1],activation='softmax'))
		self.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])



class DNNRegressor(Sequential):
	"""
	Regressor using fc nets.

	activation of hidden layer = relu
	loss = mse
	optimizer = adam
	"""

	def __init__(self, units):
		"""
		Build model. Based on specified units.
		"""
		Sequential.__init__(self)
		self.add(Dense(units[1],input_shape=(units[0],),activation='relu'))
		for i in units[2:-1]:
			 self.add(Dense(i,activation='relu'))
		self.add(Dense(units[-1]))
		self.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
