from Cell import *
from random import shuffle, randrange

class Maze:
	def __init__(self, size):
		self.size = int(size)
		self.asciiForm = self.makeRandomMaze(self.size, self.size)
		self.grid = self.convertToGrid(self.asciiForm)
	
	# Generates a random ASCII maze
	def makeRandomMaze(self, w, h):
		# From: https://rosettacode.org/wiki/Maze_generation#Python
		vis = [[0] * w + [1] for _ in range(h)] + [[1] * (w + 1)]
		ver = [["# "] * w + ['#'] for _ in range(h)] + [[]]
		hor = [["##"] * w + ['#'] for _ in range(h + 1)]
	 
		def walk(x, y):
			vis[y][x] = 1
	 
			d = [(x - 1, y), (x, y + 1), (x + 1, y), (x, y - 1)]
			shuffle(d)
			for (xx, yy) in d:
				if vis[yy][xx]: continue
				if xx == x: hor[max(y, yy)][x] = "# "
				if yy == y: ver[y][max(x, xx)] = "  "
				walk(xx, yy)
	 
		walk(randrange(w), randrange(h))
	 
		s = ""
		for (a, b) in zip(hor, ver):
			s += ''.join(a + ['\n'] + b + ['\n'])
		return s[:-2]
	
	# Converts an ASCII maze to a grid with cells
	def convertToGrid(self, asciiForm):
		# Create a list
		gridList = []
		width = 2 * self.size + 1
		for y in range(width):
			start = y * width + y
			end = (y + 1) * width + y
			gridList.append(asciiForm[start:end])
		# Create a grid
		grid = []
		for y in range(width):
			if y % 2 == 1:
				line = []
				for x in range(width):
					if x % 2 == 1:
						north = False
						if gridList[y-1][x] == "#":
							north = True
						east = False
						if gridList[y][x+1] == "#":
							east = True
						south = False
						if gridList[y+1][x] == "#":
							south = True
						west = False
						if gridList[y][x-1] == "#":
							west = True
						line.append(Cell(north, east, south, west))
				grid.append(line)
		return grid
