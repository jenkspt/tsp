#!/Users/Penn/anaconda3/bin/python3.5

'''https://googlemaps.github.io/google-maps-services-python/docs/2.4.3/'''
'''https://github.com/googlemaps/google-maps-services-python'''

import googlemaps
import time
import pandas as pd
import json

# This script uses the googlemaps api to get the distances and travel times between each city

# Use Washington DC, but Don't use Honolulu, Hawaii or Juneau, Alaska (which leaves 49 cities)
df = pd.read_csv('data/state_capitals_modified.csv', usecols=['Capital', 'Abbr'])
city_address = (df.Capital + ', ' + df.Abbr).tolist()

with open('google_api_key.json', 'r') as protected:
    p=json.loads(protected.read())

gmaps = googlemaps.Client(key=p['google_api_key'])

# You can pass a list as first argument in distance_matrix, but 49 in both throws an error
#for address in city_address:
data = gmaps.distance_matrix(city_address[0], city_address, mode='driving')
with open('data/' + city_address[0] + '.json', 'w') as outfile:
	json.dump(data, outfile)
