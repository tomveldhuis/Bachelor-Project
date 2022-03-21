from Maze import *
from QLearning import *
from DoubleQLearning import *

def main():
	size = 4
	initValue = 0
	maze = Maze(size)
	print(maze.asciiForm)
	QLearning(maze)
	printQTable(maze)
	DoubleQLearning(maze)
	printDoubleQTable(maze)
	
if __name__ == "__main__":
	main()
