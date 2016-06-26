#!/Users/Penn/anaconda3/bin/python3.5

import pandas as pd
import numpy as np
import json

df = pd.read_csv('data/state_capitals_modified.csv', usecols=['Capital', 'Abbr'])

# Lookup tables for finding the travel time and distances between cities.
# The rows (first index) corresponse with the origin and
# the columns (second index) corresponds with the destination
# while the value is either the travel time in seconds or distance in meters.
# For the traveling salesman problem we will only be using the travel time
time_table = np.zeros([49,49])
distance_table = np.zeros([49,49])

df['Address'] = df.Capital + ', ' + df.Abbr
for row, origin in enumerate(df.Address):
	with open('data/city_travel_times/' + origin + '.json') as infile:
		data = json.load(infile)

	if data['status'] == 'OK':
		for col, destination in enumerate(data['rows'][0]['elements']):
			if destination['status'] == 'OK':
				time_table[row,col] = destination['duration']['value']
				distance_table[row,col] = destination['distance']['value']
			else:
				print('\tFAIL:', 'col', col)
	else:
		print('FAIL:', 'row', row)

# Save the tables to binary files
np.save('data/time_lookup_table.npy', time_table)
np.save('data/distance_lookup_table.npy', distance_table)

#:::: DATA VERIFICATION ::::

# Each repeated index for rows and columns should be zero because
# it is the distance from the given city to itself
# the table should follow this pattern:
# 0.0 val val val val
# val 0.0 val val val
# val val 0.0 val val
# val val val 0.0 val
# val val val val 0.0 
# val val val val 0.0 
# val val val 0.0 val
# val val 0.0 val val
# val 0.0 val val val
# 0.0 val val val val

"""
for index in range(49):
	print(str(index)+','+str(index), time_table[index,index])
for index in range(49,98):
	row = index
	col = index - 49
	print(str(row)+','+str(49-col-1), time_table[row,-col])
"""
#for row in time_table:
#	print(' '.join(str(i) for i in row))

"""
print('Spot Check')
print('time 12,19', time_table[12,19])
print('distance 12,19', distance_table[12,19])
print('time 44,13', time_table[44,13])
print('distance 44,13', distance_table[44,13])
print('time 23,11', time_table[23,11])
print('distance 23,11', distance_table[23,11])
"""