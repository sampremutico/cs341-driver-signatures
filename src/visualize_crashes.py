from matplotlib import pyplot as plt

def plot_course(df):
  x = []
  y = []
  crashes_x1 = []
  crashes_y1 = []
  crashes_x2 = []
  crashes_y2 = []
  for idx, row in df.iterrows():
    if idx % 500 == 0:
      x.append(row['POSITION_X'])
      y.append(row['POSITION_Y'])
      crashes_x1.append(row['COLLISION_LOCATION_X_1'])
      crashes_y1.append(row['COLLISION_LOCATION_Y_1'])
      # crashes_x2.append(row['COLLISION_LOCATION_X_2'])
      # crashes_y2.append(row['COLLISION_LOCATION_Y_2'])

  plt.scatter(x, y, color='blue')
  plt.scatter(crashes_x1, crashes_y1, color='red')
  # plt.scatter(crashes_x2, crashes_y2, color='green')

  plt.show()
