from Maze import *
from QLearning_Form import *

from NeuralNetwork import *

def test():
	nn = NeuralNetwork(5)
	print(nn.predict(0, 0))
	#nn.update(0, 0, "east", 100)
	#print(nn.predict(0, 0))

def main():
	test()
	
if __name__ == "__main__":
	main()
