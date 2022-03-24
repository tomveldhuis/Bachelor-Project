from Maze import *
from QLearning import *
from DoubleQLearning import *

def test():
	size = 4
	initValue = 0
	maze = Maze(size)
	print(maze.asciiForm)
	# QLearning(maze)
	# printQTable(maze)
	DoubleQLearning(maze)
	printDoubleQTable(maze)

def main():
	test()
	
if __name__ == "__main__":
	main()
