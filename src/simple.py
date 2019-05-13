import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from datetime import datetime
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

DATA = "../data/nervtech_v1_drives-no-collisions_user_1634_scenario_0_repeat_0_opti.csv"

def prep_data(file=DATA, label='SPEED_LIMIT', split=0.9):
    df = pd.read_csv(file)
    df = df.drop(['DATE'], axis=1)
    print(df.columns.values)
    # Encoding class labels
    class_to_predict = label
    class_mapping = {label:idx for idx,label in enumerate(np.unique(df[class_to_predict]))}
    #for idx, label in class_mapping.items():
    #    print("Class: {} --> {} kmph Speed Limit".format(str(label), str(idx)))
    df[class_to_predict] = df[class_to_predict].map(class_mapping)

    X = df.drop([class_to_predict], axis=1).values
    y = df[class_to_predict].values

    timestamp = df["TIMESTAMP"]
    collisions_x1 = df["COLLISION_LOCATION_X_1"]
    #collisions_y1 = df["POSITION_X"]
    collision_entity_1 = df["COLLISION_ID_1"]
    collision_entity_2 = df["COLLISION_ID_2"]

    plt.scatter(timestamp, collisions_x1,s=2)
    plt.scatter(timestamp, collision_entity_1,s=2)
    plt.scatter(timestamp, collision_entity_2,s=2)
    #plt.scatter(timestamp, collisions_y1)
    plt.legend()
    plt.show()

    generate_sequences(df)

    train_len = int(0.8 * len(y))
    test_len = int(len(y) - train_len)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = split)
    return X_train, X_test, y_train, y_test

# Takes in a drive and returns sequences of window_size_seconds*30,1
def generate_sequences(drive, window_size_seconds=5, prediction_window_seconds=2):

    collisions = list()
    prev_row = None#drive.iloc(0)
    #print(prev_row["COLLISION_LOCATION_X_1"])
    for index, row in drive.iterrows():
        if index != 0 and np.abs(row["COLLISION_LOCATION_X_1"] - prev_row["COLLISION_LOCATION_X_1"]) > 100:
            ts = row["TIMESTAMP"] / 1000
            formatted = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S.%f')
            collisions.append((formatted, row["COLLISION_LOCATION_X_1"]))
            #print("Found Collision")
        prev_row = row
    print("Found {} collisions".format(len(collisions)))
    print("FOund {} uniqiue collisions".format(len(set(collisions))))
    for collision in collisions:
        print(collision)



class DynamicNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(DynamicNet, self).__init__()
        print("\nInput dimensions --> {}\n# Classes --> {}".format(D_in, D_out))
        self.input_linear = torch.nn.Linear(D_in, D_out)
        torch.nn.init.xavier_uniform_(self.input_linear.weight)
        self.batch_norm = torch.nn.BatchNorm1d(D_out)
        #self.middle_linear = torch.nn.Linear(H, H)
        #self.output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_lin = self.input_linear(x)
        h_lin = self.batch_norm(h_lin)
        #for _ in range(random.randint(0, 3)):
        #h_relu = self.middle_linear(h_relu).clamp(min=0)
        y_pred = h_lin
        return y_pred

def run(data, class_to_predict, split_size, learning_rate=0.5, num_epochs=500, batch_size=50):

    X_train, X_test, y_train, y_test = prep_data(data, class_to_predict, split_size)
    exit(0)
    num_classes = len(set(y_train.tolist()))
    D_in, H, D_out = X_train.shape[1], 4, num_classes
    model = DynamicNet(D_in, H, D_out)

    if torch.cuda.is_available():
        print('CUDA Available')
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        if torch.cuda.is_available():
            x = Variable(torch.Tensor(X_train).cuda())
            y = Variable(torch.Tensor(y_train).cuda())
        else:
            x = Variable(torch.Tensor(X_train).float())
            y = Variable(torch.Tensor(y_train).long())

        current_batch = 0
        for iteration in range(y.shape[0] // batch_size):
            batch_x = x[current_batch: current_batch + batch_size]
            batch_y = y[current_batch: current_batch + batch_size]
            current_batch += batch_size
            #batch_x, batch_y
            #print(len(y), len(x),batch_y, current_batch, current_batch+batch_size)
            y_pred = model(batch_x)

            loss = criterion(y_pred, batch_y)

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (iteration == 0) and ((epoch) % 100 == 0):

                #for name, param in model.named_parameters():
                #    print(name, param)
                print("Predictions: ", y_pred[0].data)
                print('Epoch [%d/%d] Loss: %.4f' %(epoch + 1, num_epochs, loss.data))

    if torch.cuda.is_available():
        x = Variable(torch.Tensor(X_test).cuda())
        y = torch.Tensor(y_test).long()
    else:
        x = Variable(torch.Tensor(X_test).float())
        y = torch.Tensor(y_test).long()

    out = model(x)

    _, predicted = torch.max(out.data, 1)

    print("\nExpected counts:")
    y_list = y.tolist()
    for v in set(y_list):
        print(v, y_list.count(v))

    print("\nFound counts:")
    pred_list = predicted.tolist()
    for v in set(pred_list):
        print(v, pred_list.count(v))

    print('Accuracy of the network %d %%' % (100 * torch.sum(y==predicted) / (1.0*len(y_test))))
    print("\nConfusin matrix:")
    print(confusion_matrix(y,predicted))

if __name__ == "__main__":
    run(data=DATA, class_to_predict='SPEED_LIMIT', split_size=0.1, learning_rate=0.5, num_epochs=5)