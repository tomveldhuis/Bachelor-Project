from QLearning_Meta import parameters

from QLearning import NormalQLearning, printNormalQTable
from DoubleQLearning import DoubleQLearning, printDoubleQTable

def QLearning(maze, form = "normal"):
	if form == "normal":
		NormalQLearning(maze, parameters())
	elif form == "double":
		DoubleQLearning(maze, parameters())

def printQTable(maze, form):
	if form == "normal":
		printNormalQTable(maze)
	elif form == "double":
		printDoubleQTable(maze)
