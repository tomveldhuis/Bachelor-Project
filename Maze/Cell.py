class Cell:
	def __init__(self, initValue, north, east, south, west):
		self.walls = {
		"north": bool(north),
		"east": bool(east),
		"south": bool(south),
		"west": bool(west)
		}
		self.values = {
		"north": float(initValue),
		"east": float(initValue),
		"south": float(initValue),
		"west": float(initValue)
		}
