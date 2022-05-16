from keras.models import Model
from keras.layers import Dense, Input
import numpy as np
import os

class NeuralNetwork:
	# Constructor
	def __init__(self, size):
		self.model = self.createNetwork(size)
	
	def createNetwork(self, size):
		inputs = Input(shape=(2,), name="input")
		hidden = Dense(10, activation='relu', name="hidden")(inputs)
		outputs = Dense(4, activation='relu', name="output")(hidden)
		
		out1 = Dense(1, activation = "linear", name = "out1")(hidden)
		out2 = Dense(1, activation = "linear", name = "out2")(hidden)
		out3 = Dense(1, activation = "linear", name = "out3")(hidden)
		out4 = Dense(1, activation = "linear", name = "out4")(hidden)
		
		model = Model(inputs=inputs, outputs=outputs, name="NeuralTable")
		model.compile(loss = 'mean_squared_error', optimizer = 'sgd')
		print(model.summary())
		return model
	
	def predict(self, x, y):
		data = self.model.predict([(x, y)])[0]
		dict = {
		"north": data[0],
		"east": data[1],
		"south": data[2],
		"west": data[3]
		}
		return dict
	
	def update(self, x, y, action, value):
		# Determine weights
		weights = {0:0, 1:0, 2:0, 3:0}
		if action == "north":
			weights[0] = 1
		elif action == "east":
			weights[1] = 1
		elif action == "south":
			weights[2] = 1
		elif action == "west":
			weights[3] = 1
		else:
			return
		
		print(weights)
		print(value)
		
		# Create mapping data
		input = np.array([[x, y]])
		target = np.array([[value, value, value, value]])
		
		self.model.fit(input, target, class_weight = weights, batch_size = 32, epochs = 1)
		
		
	
	
	
