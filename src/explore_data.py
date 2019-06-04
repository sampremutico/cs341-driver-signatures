from utils import load_numpy_data
import numpy as np
import json
from matplotlib import pyplot as plt

# unnormalize if you want to look at actual differences
X_train, X_val, y_train, y_val = load_numpy_data(5, (5, 10), normalize=True)


# num collisions, num non
X = np.concatenate([X_train, X_val], axis=0)
y = np.concatenate([y_train, y_val], axis=0)

print('Total segments: {}'.format(X.shape[0]))
X_collisions = X[y == 1]
y_collisions = y[y == 1]

X_none= X[y == 0]
y_none = y[y == 0]

print('Total collisions segments (ie, num crashes): {}'.format(X_collisions.shape[0]))
print('Total no collisions segments: {}'.format(X_none.shape[0]))

X_collisions_avg_dims = np.mean(X_collisions, axis=(0, 1))
X_none_avg_dims = np.mean(X_none, axis=(0, 1))
X_collisions_std_dims = np.std(X_collisions, axis=(0, 1))
X_none_std_dims = np.std(X_none, axis=(0, 1))




# average values in each
with open('idx_to_column_names.json') as f:
  col_dict = json.loads(f.read())

for col_idx in col_dict.keys():
  col_name = col_dict[col_idx]
  col_idx = int(col_idx)

  print(col_name)
  print('Collision')
  print('mean: {} std: {}'.format(X_collisions_avg_dims[col_idx], X_collisions_std_dims[col_idx]))
  print('No Collision')
  print('mean: {} std: {}'.format(X_none_avg_dims[col_idx], X_none_std_dims[col_idx]))
  print('')


#TODO: interesting variables that look different (formalize this)

# road angle
# speed next vehicle
# acceleration pedal
# brake pedal
# curve radius
# acceleration_next_vehicle
# acceleration
# speed
# fog
# speed limit
# time of driving
# distance to next stop signal
# fast lane

# If it's normalized we actually don't need error bars because its 1
collision_bar_means = []
collision_bar_stds = []
none_bar_means = []
none_bar_stds = []

with open('column_names_to_idx.json') as f:
  col_dict = json.loads(f.read())

# interesting_vars = ['ROAD_ANGLE', 'SPEED_NEXT_VEHICLE', 'ACCELERATION_PEDAL', 'BRAKE_PEDAL', 'CURVE_RADIUS', 'ACCELERATION_NEXT_VEHICLE', 'ACCELERATION', 'SPEED', 'FOG', 'SPEED_LIMIT', 'TIME_OF_DRIVING', 'DISTANCE_TO_NEXT_STOP_SIGNAL', 'FAST_LANE']

interesting_vars = ['ROAD_ANGLE',  'CURVE_RADIUS',  'FOG',  'TIME_OF_DRIVING', 'DISTANCE_TO_NEXT_STOP_SIGNAL', ]
labels = ['rd angle', 'brk pedal', 'rd curve', 'clst car acc', 'acc', 'fog', 'speed limit', 'time ', 'dst to signal', 'fast lane']

interesting_vars = ['ACCELERATION', 'ACCELERATION_NEXT_VEHICLE', 'BRAKE_PEDAL', 'SPEED_LIMIT', 'ROAD_ANGLE', 'CURVE_RADIUS', 'FOG', 'TIME_OF_DRIVING', 'DISTANCE_TO_NEXT_STOP_SIGNAL']
labels = ['acc', 'acc clst car', 'brk pedal', 'spd limit', 'rd angle', 'rd curve', 'fog', 'time', 'dst to signal']


for var in interesting_vars:
  col_idx = int(col_dict[var])
  collision_bar_means.append(X_collisions_avg_dims[col_idx])
  collision_bar_stds.append(X_collisions_std_dims[col_idx])
  none_bar_means.append(X_none_avg_dims[col_idx])
  none_bar_stds.append(X_none_std_dims[col_idx])

barWidth = 0.25

# Set position of bar on X axis
r1 = np.arange(len(collision_bar_means))
r2 = [x + barWidth for x in r1]

# Make the plot
plt.bar(r1, collision_bar_means, width=barWidth, edgecolor='white', label='collision sequences')
plt.bar(r2, none_bar_means, width=barWidth, edgecolor='white', label='no collision sequences')

# Add xticks on the middle of the group bars
plt.xticks([r + barWidth for r in range(len(interesting_vars))], labels, fontsize=8)

axes = plt.gca()
axes.set_ylim([-0.15,0.25])



# Create legend & Show graphic
plt.legend()
plt.show()





# make some plots

