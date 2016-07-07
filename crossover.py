import numpy as np

'''http://www.rubicite.com/Tutorials/GeneticAlgorithms/CrossoverOperators/PMXCrossoverOperator.aspx/'''

def pmx(p1, p2):
	if len(p1) != len(p1):
		raise ValueError('dimension mismatch')

	n = len(p1)
	p2_arg = p2.argsort()
	#swath_size = np.random.randint(2,n-1)
	swath_size = 5
	x1 = np.random.randint(n+1-swath_size)
	x2 = x1 + swath_size
	child = np.full(n, -1, dtype='int64')
	child[x1:x2] = p1[x1:x2]
	swath = child[x1:x2]
	for i in range(x1,x2):
		if not p2[i] in swath:

			p2_i = p2_arg[p1[i]]
			while x1 <= p2_i and x2 > p2_i:
				p2_i = p2_arg[p1[p2_i]]
			child[p2_i] = p2[i]

	for i in range(0, n):
		if child[i] == -1:
			child[i] = p2[i]		
	return child

if __name__ == "__main__":
	p1 = np.array([8, 4, 7, 3, 6, 2, 5, 1, 9, 0])
	p2 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
	c1 = np.array([0, 7, 4, 3, 6, 2, 5, 1, 8, 9])

	result = pmx(p1, p2)

	print('Parent 1: ', p1)
	print('Parent 2: ', p2)
	print('Result  : ', result)
	print('Correct?: ', c1==result)
