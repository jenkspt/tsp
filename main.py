import numpy as np
import time
import selection
import crossover

np.random.seed(int(time.time()))
# n is number of cities
# p_size is the desired poulation size
n = 49
p_size = 1000
generations = 1000
dtype='int8'

pop = np.ndarray((n + 1, p_size), dtype=dtype)
# First and last row are locked in as n-1 (this way when they are removed the indexing doesn't change)
pop[[0,-1],:] = n-1
# Create random population
# Each city is encoded as an integer from 0 to 48
for col in range(pop.shape[1]):
	pop[1:-1,col] = np.random.permutation(n-1)
# Look-up table for distances between cities
lut = np.load('data/distance_lookup_table.npy')


for g in range(generations):

	fitness = lut[ (pop[:-1,], pop[1:,]) ].sum(axis=0)
	weights = selection.weights(fitness)
	weights_sum = weights.sum()

	# Found empirically that selecting p_size/3.5 parents results
	# in a population roughly the same size
	# 2-D array of chromosomes selected for crossover
	winners = np.ndarray((2, int(p_size/3.5)), dtype='int64')
	for i in range(winners.shape[1]):
		winners[0,i] = selection.fps(weights, weights_sum)	# Parent 1
		winners[1,i] = selection.fps(weights, weights_sum)	# Parent 2

	"""
	# np.unique(winners).size-1 is the number of parents in the new population
	# winners.size represents the number of offspring in the new population
	print('Uniqe parents', np.unique(winners).size-1)
	print('Proportion', (np.unique(winners).size-1)/p_size)
	print('Offspring', winners.size)
	print('Proportion', winners.size/p_size)
	print('Total proportion', (np.unique(winners).size-1 + winners.size)/p_size)
	"""
	# Number of parents that will reproduce
	unique_winners = np.unique(winners)
	losers = np.ones(pop.shape[1])
	losers[unique_winners] = False
	losers = np.nonzero(losers)[0]		# Get the indicese of the chromosomes that don't survive
	i, pi = 0, 0
	while (i < losers.size-1) and pi < winners.shape[1]:
		p1 = pop[1:-1, winners[0, pi]]
		p2 = pop[1:-1, winners[1, pi]]
		pop[1:-1,losers[i]] = crossover.pmx(p1, p2)
		pop[1:-1,losers[i+1]] = crossover.pmx(p2, p1)
		i += 2
		pi += 1

	"""
	distances = lut[ (pop[:-1,], pop[1:,]
	weights = selection.weights() ].sum(axis=0) )
	weights_sum = weights.sum()
	"""
	if g%100 == 0:
		stats = [g, fitness.max()/1000, fitness.mean()/1000, fitness.min()/1000]
		print('Generation:{:.0f}, Max:{:.0f}, Average:{:.0f}, Min:{:.0f}'.format(*stats))

print(pop[:,fitness.argmax()])