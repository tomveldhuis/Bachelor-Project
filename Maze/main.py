from Maze import *
from QLearning import *

def main():
	size = 5
	initValue = 0
	maze = Maze(size, initValue)
	print(maze.asciiForm)
	QLearning(maze)
	printQTable(maze)
	
if __name__ == "__main__":
	main()
