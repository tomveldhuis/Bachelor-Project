class Cell:
	def __init__(self, north, east, south, west):
		self.walls = {
		"north": bool(north),
		"east": bool(east),
		"south": bool(south),
		"west": bool(west)
		}
