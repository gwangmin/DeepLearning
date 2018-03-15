from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import pylab as plt

class DNN(object):
	# Fully Connected Network.
	#
	# units: List. [input dim, hidden unit1, hidden unit2, ... , output dim]
	# last_activation: Output layer's activation function.
	# optimizer: Optimizer. Defauld Adam(lr=0.01).
	def __init__(self, units, last_activation, loss='mse', optimizer=Adam(lr=0.01)):
		self.model = self.build(units, last_activation, loss, optimizer)

	# Build network
	# parameters == __init__ parameters
	def build(self, units, last_activation, loss, optimizer):
		model = Sequential()
		model.add(Dense(units[1], input_dim=units[0], activation='relu', kernel_initializer='he_uniform'))
		for unit in units[2:-1]:
			model.add(Dense(unit, activation='relu', kernel_initializer='he_uniform'))
		model.add(Dense(units[-1], activation=last_activation, kernel_initializer='he_uniform'))
		model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

	# Predict
	#
	# x: Network input
	def predict(self, x):
		return self.model.predict(x)

	# Fit
	#
	# x: Network input.
	# y: Label.
	def fit(self, x, y):
		self.model.fit(x, y)

	# Save weights
	#
	# path: Path.
	def save(self, path):
		self.model.save_weights(path)

	# Load weights
	#
	# path: Path
	def load(self, path):
		self.model.load_weights(path)

	# Return accuracy
	#
	# x: Network input.
	# y: Label.
	def	get_accuracy(self, x, y):
		return model.evaluate(x, y)[1]

