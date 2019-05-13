
import pandas as pd
import numpy as np
from tqdm import tqdm

path_name = '../data/cs341-driver-data/nervtech/v1/drives-with-collisions/user_1594_scenario_3_repeat_0_opti.csv'

# not clear how many collisions there are -- different number of x and y values...
# and what do collision 1 and 2 signify? we have no idea?

data = pd.read_csv(path_name)

# print(data['COLLISION_LOCATION_X_1'].unique()[:20])
# print(len(data['COLLISION_LOCATION_X_1'].unique()))
# print(len(data['COLLISION_LOCATION_Y_1'].unique()))
# print(len(data['COLLISION_LOCATION_X_2'].unique()))
# print(len(data['COLLISION_LOCATION_Y_2'].unique()))


locations = []

# we are saying that the collision has to have taken place at least
# 50 units away from previous collision

for index, row in tqdm(data.iterrows()):
  if index == 0: continue
  x, y = float(row['COLLISION_LOCATION_X_1']), float(row['COLLISION_LOCATION_Y_1'])
  if len(locations) == 0:
    locations.append((x, y))
    continue
  prev_x, prev_y = locations[-1]
  if np.abs(prev_x - x) > 50 or np.abs(prev_y - y) > 50:
    locations.append((x, y))

print(len(locations))
print(locations)

locations = []

# we are saying that the collision has to have taken place at least
# 50 units away from previous collision

