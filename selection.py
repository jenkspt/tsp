import numpy as np

def weights(fitness):
	min_val, max_val = fitness.min(), fitness.max()
	weights = np.ndarray(fitness.size)
	for i in range(weights.size):
		# Invert the probabilites (because we want to minimize the distance)
		weights[i] = np.interp(abs(fitness[i]-max_val), [0, 1], [min_val, max_val])
	return weights

def fps(weights, weight_sum):
	value = np.random.rand() * weight_sum
	for i, weight in enumerate(weights):		
		value -= weight
		if value <= 0:
			return i
	return len(weights) - 1