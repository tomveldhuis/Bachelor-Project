from Maze import *
from QLearning_Form import *

def test():
	size = 4
	initValue = 0
	maze = Maze(size)
	print(maze.asciiForm)
	
	form = "double"
	QLearning(maze, form)
	printQTable(maze, form)

def main():
	test()
	
if __name__ == "__main__":
	main()
