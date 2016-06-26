import numpy as np
import time
import matplotlib.pyplot as plt

def random_population(n, p_size, dtype='uint8'):
	# n is number of cities
	# p_size is the desired poulation size
	p = np.zeros((n + 1, p_size), dtype=dtype)
	for col in range(p_size):
		p[1:-1,col] = np.random.permutation(n-1) + 1
	return p

def get_travel_time(p, lut):
	return lut[ (p[:-1], p[1:,]) ]

def tournament_select(f, k=2, c=0.5, dtype='uint64'):
	# f is the fitness for each element in the population (1d array)
	# k is the size of the group in the selection; f.size should be a multiple of k
	# c is the proportion of the population to select: 0 < c <= 1

	num_groups = int(f.shape[0]/k)
	num_winners = int(c * k) # number of winners in each group
	# Create 2d array of random indicese with each column being a list of groups
	groups_index = np.reshape(np.random.permutation(f.shape[0]), (k, num_groups))
	# Explain some shit
	group = f[groups_index]
	group_argsort = np.argsort(group, axis=0)[:num_winners,:]
	winners = groups_index[group_argsort,np.arange(group.shape[1])].flatten().astype(dtype)
	#Since the winners are shuffled, we can just select the adjacent lists for crossover
	np.random.shuffle(winners)
	return(winners)

def crossover(pw, xover_prop=0.25, dtype='uint8'):
	# Subtract 1 from all elements to account for the 0th index being removed.
	# Need to add this back when reconstructing the new population 
	p = pw[1:-1,:]-1
	pi = np.argsort(p, axis=0)					# Array of sorted indicese
	half = int(p.shape[1]/2)

	# Split the 2d arrays in half column-wise and stack them in 3d
	w_values = np.stack( (p[:,:half], p[:,half:half*2]), axis=-1 )
	w_indexes = np.stack( (pi[:,:half], pi[:,half:half*2]), axis=-1)
	# Since each city has to actually swap places with another city
	# an xover_prop of 0.25 will actually change 1/2 of the list
	xover_num = int(xover_prop * (w_values.shape[0]))

	swap_list = np.ndarray((xover_num, w_values.shape[1]), dtype=dtype)
	columns = np.arange(w_values.shape[1])
	# Randomly generated indicese to be used for cross-over
	for col in columns:
		swap_list[:,col] = np.random.permutation(w_values.shape[0])[:xover_num]

	# Get the values from the random list of indicies for parent1 and parent2
	# parent 1 and parents 2 are represented as stacked planes in 3d.
	parent1_vals = w_values[swap_list, columns, 0]
	parent2_vals = w_values[swap_list, columns, 1]

	# get the location (index) of the values from parent1 in parent2.
	parent1_swap_ind = w_indexes[parent2_vals, columns, 0]
	parent2_swap_ind = w_indexes[parent1_vals, columns, 1]

	#print(parent1_swap_ind.shape)
	# Assign the values looked up in the previous step with the values looked up
	# using the swap_list.
	w_values[parent1_swap_ind, columns, 0] = parent1_vals
	w_values[parent2_swap_ind, columns, 1] = parent2_vals

	# Assign the values from parent1 to parent2 and vice-versa
	w_values[swap_list, columns, 0] = parent2_vals
	w_values[swap_list, columns, 1] = parent1_vals

	offspring = np.zeros((w_values.shape[0]+2, w_values.shape[1]*w_values.shape[2]), dtype=dtype)
	# Return the results of the crossover to a 2d array, add 1 and add zeros to the first and last row
	#offspring =  np.reshape(w_values, (w_values.shape[0], w_values.shape[1]*w_values[2]))
	offspring[1:-1,:] =  w_values.reshape((w_values.shape[0], w_values.shape[1]*w_values.shape[2]))+1
	return offspring

def mutate(p, mut_prop=.02, dtype='uint8'):
	# Chromosome(columns) to be mutated
	columns = np.where( np.random.rand(p.shape[1]) <= 0.02 )[0]
	# row position of cities to be swapped
	swap1 = np.random.randint(1,48, columns.shape[0])
	swap2 = np.random.randint(1,48, columns.shape[0])
	temp = p[swap1, columns]
	p[swap1, columns] = p[swap2, columns]
	p[swap1, columns] = temp




def generate(p, lut, g):
	np.random.seed(int(time.time()))
	t = get_travel_time(p, lut)
	f = np.sum(t, axis=0)			# Fitness

	winners = tournament_select(f, k=10, c=0.5, dtype='uint64')
	losers = np.ones(p.shape[1], dtype=bool)
	# Use boolean filter to remove all of the losers and replace them with new offspring
	losers[winners] = False
	p[:,losers] = crossover(p[:,winners])	#offspring
	mutate(p, 0.02)
	if g%25 == 0:
		print()
		#print('WINNERs: Average:{:.0f}, Max:{:.0f}, Min:{:.0f}'.format(np.mean(fw), np.amax(fw), np.amin(fw)))
		#print('LOSERS: Average:{:.0f}, Max:{:.0f}, Min:{:.0f}'.format(np.mean(fl), np.amax(fl), np.amin(fl)))
		print('Generation:{:.0f}\n\tAverage:{:.0f}, Max:{:.0f}, Min:{:.0f}'.format(g, f.mean()/1000, f.max()/1000, f.min()/1000))
		return [f.mean(), f.max(), f.min()]

np.random.seed(int(time.time()))
n = 49
p_size = 10000
gs = 500
p = random_population(n, p_size)
lut = np.load('data/distance_lookup_table.npy')

track_result = np.ndarray((gs,3))
for g in range(gs):
	track_result[g] = generate(p, lut, g)

t = get_travel_time(p, lut)
winner = np.argmax(np.sum(t, axis=0))
print(p[:,winner])






	
if __name__ == "__main__":
    # execute only if run as a script
    """
	n = 49
	p_size = 500000
	p = None
	p = random_population(n, p_size)
	print(p[:,8:10])
	"""
	#np.load('data/time_lookup_table.npy', time_table)
