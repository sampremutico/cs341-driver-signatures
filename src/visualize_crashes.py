from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


#TODO: Move these to utils
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

  plt.scatter(x, y)
  plt.scatter(crashes_x1, crashes_y1, label='crashes')
  plt.legend()
  # plt.scatter(crashes_x2, crashes_y2, color='green')
  plt.show()
  plt.savefig('course_with_crashes')


# plot_course(pd.read_csv('../data/cs341-driver-data/nervtech/v1/drives-with-collisions/user_1636_scenario_0_repeat_0_opti.csv'))



def plot_loss_functions(filename):
  with open(filename) as f:

    arr = np.linspace(.5, 20, 40)

    lines = [l.split() for l in f.readlines()[3:]]
    train_losses = [float(l[0]) for l in lines]
    plt.plot(arr, train_losses)
    # plt.xticks(np.arange(0, 21, step=4))
    plt.ylabel('Train Loss')
    plt.xlabel('Epoch')



    plt.show()


    val_f1s = [float(l[1]) for l in lines]
    train_f1s = [float(l[5]) for l in lines]
    plt.plot(arr, train_f1s, label='Train F1')
    plt.plot(arr, val_f1s, label='Val F1')
    # plt.xticks(np.arange(1, 21, step=4))
    plt.ylabel('F1 Score')
    plt.xlabel('Epoch')

    plt.legend()
    plt.show()



# plot_loss_functions('experiments/seq_len_5_window_5_10/lstm/0.3437449')


plot_loss_functions('experiments/seq_len_5_window_5_10/lstm/0.39559932')

# plot_loss_functions('experiments/seq_len_5_window_5_10/lstm/0.4')