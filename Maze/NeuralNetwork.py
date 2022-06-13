from keras.models import Model
from keras.layers import Dense, Input
import numpy as np
import os

from random import random

class NeuralNetwork:
	# Constructor
	def __init__(self, size):
		self.model = self.createNetwork(size)
	
	def createNetwork(self, size):
		inputs = Input(shape=(2,), name="inputs")
		hidden = Dense(size, activation="relu", name="hidden")(inputs)
		outputs = Dense(4, activation="relu", name="outputs")(hidden)
		
		model = Model(inputs=inputs, outputs=outputs, name="QTable")
		model.compile(loss="mse", optimizer="sgd")
		return model
	
	def predict(self, x, y):
		prediction = self.model.predict([(x, y)])
		# Choose the prediction based on epsilon greedy
		
		# Perform the action from epsilon greedy
		
		# Get the new state and reward from the environment
		
		# Get the prediction for the new state and take the maximum
		
		# Calculate the target value
		
		model.fit((x, y), targetValue)
		return prediction[0]
		
		
		
	
	
	
