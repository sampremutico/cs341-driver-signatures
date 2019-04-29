import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

DATA = "../data/cs341-driver-data/nervtech/v1/drives-with-collisions/user_1636_scenario_0_repeat_0_opti.csv"


def prep_data(file=DATA, label='SPEED_LIMIT', split=0.9):
    df = pd.read_csv(file)
    df = df.drop(['DATE'], axis=1)

    # Encoding class labels
    class_to_predict = label
    class_mapping = {label:idx for idx,label in enumerate(np.unique(df[class_to_predict]))}
    print(class_mapping)
    df[class_to_predict] = df[class_to_predict].map(class_mapping)

    X = df.drop([class_to_predict], axis=1).as_matrix()
    y = df[class_to_predict].as_matrix()

    train_len = int(0.8 * len(y))
    test_len = int(len(y) - train_len)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = split)
    return X_train, X_test, y_train, y_test

class DynamicNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(DynamicNet, self).__init__()
        #self.input_linear = torch.nn.Linear(D_in, H)
        print("Input dimensions: {}\n# Classes: {}".format(D_in, D_out))
        self.input_linear = torch.nn.Linear(D_in, D_out)
        self.tanh = torch.nn.Tanh()
        #self.middle_linear = torch.nn.Linear(H, H)
        #self.output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_lin = F.relu(self.input_linear(x))
        #for _ in range(random.randint(0, 3)):
        #h_relu = self.middle_linear(h_relu).clamp(min=0)
        y_pred = h_lin
        return y_pred

def run(data, class_to_predict, split_size, learning_rate=0.05, num_epochs=500): 
    
    X_train, X_test, y_train, y_test = prep_data(data, class_to_predict, split_size)

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

        y_pred = model(x)
        #for param in model.parameters():
        #    print(param.data)

        loss = criterion(y_pred, y)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch) % 100 == 0:
            print(x[0])
            print(y_pred[0])
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
    run(data=DATA, class_to_predict='SPEED_LIMIT', split_size=0.1, learning_rate=0.5, num_epochs=500)