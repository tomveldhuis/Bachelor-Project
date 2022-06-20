from Maze import *
from QLearning_Form import *
from DeepQLearning import *

def test():
	experiment(parameters(), standardMazes())
	
def main():
	test()
	
if __name__ == "__main__":
	main()
