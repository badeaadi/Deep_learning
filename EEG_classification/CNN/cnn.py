import torch.nn.functional as F
import torch.nn as nn
import torch


class Args():
    def __init__(self):

        self.batch_size = 512
        self.epochs = 50
        self.lr = 5e-4
        self.momentum = 0.9
        self.seed = 42
        self.log_interval = int(8000 / self.batch_size)
        self.cuda = True


in_channels = 23
no_filters1 = 50
no_filters2 = 100
no_filters3 = 150
no_filters4 = 200

no_neurons1 = 2048
no_neurons2 = 512
no_neurons3 = 32
out_features = 2

in_features = int(4684800/512)


class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=no_filters1, kernel_size=5, stride=1)
        self.batch1 = nn.BatchNorm1d(no_filters1)
        self.conv2 = nn.Conv1d(in_channels=no_filters1,
                               out_channels=no_filters2, kernel_size=5, stride=1)
        self.batch2 = nn.BatchNorm1d(no_filters2)

        self.conv3 = nn.Conv1d(in_channels=no_filters2,
                               out_channels=no_filters3, kernel_size=3, stride=1)
        self.batch3 = nn.BatchNorm1d(no_filters3)

        self.fc1 = nn.Linear(in_features=in_features, out_features=no_neurons1)
        self.fc2 = nn.Linear(in_features=no_neurons1, out_features=no_neurons2)
        self.fc3 = nn.Linear(in_features=no_neurons2, out_features=no_neurons3)
        self.fc4 = nn.Linear(in_features=no_neurons3,
                             out_features=out_features)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        # x = F.max_pool1d(x, 2)
        x = self.batch1(x)
        x = F.dropout(x, p=0.15)

        # print(x.shape)

        x = F.relu(self.conv2(x))
        x = self.batch2(x)
        x = F.max_pool1d(x, 2)
        x = F.dropout(x, p=0.15)

        x = F.relu(self.conv3(x))
        x = self.batch3(x)
        x = F.max_pool1d(x, 2)
        x = F.dropout(x, p=0.15)

        x = x.view(-1, in_features)

        # print(self.fc1, x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return F.log_softmax(x, dim=1)
