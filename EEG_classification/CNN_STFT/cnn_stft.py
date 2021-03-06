import torch.nn.functional as F
import torch.nn as nn
import torch


class Args():
    def __init__(self):

        self.batch_size = 512
        self.epochs = 25
        self.lr = 1e-6
        self.momentum = 0.9
        self.seed = 42
        self.log_interval = int(8000 / self.batch_size)
        self.cuda = True
        self.weight_decay = 1e-6


in_channels = 23
no_filters1 = 200
no_filters2 = 250
no_filters3 = 150
no_filters4 = 200
no_filters4 = 300

no_neurons1 = 2048
no_neurons2 = 512
no_neurons3 = 256
no_neurons4 = 128
out_features = 2

in_features = int(2304000/512)


class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=no_filters1, kernel_size=7, stride=1)
        self.batch1 = nn.BatchNorm1d(no_filters1)
        self.conv2 = nn.Conv1d(in_channels=no_filters1,
                               out_channels=no_filters2, kernel_size=5, stride=1)
        self.batch2 = nn.BatchNorm1d(no_filters2)

        self.conv3 = nn.Conv1d(in_channels=no_filters2,
                               out_channels=no_filters3, kernel_size=5, stride=1)
        self.batch3 = nn.BatchNorm1d(no_filters3)

        self.conv4 = nn.Conv1d(in_channels=no_filters3,
                               out_channels=no_filters4, kernel_size=3, stride=1)
        self.batch4 = nn.BatchNorm1d(no_filters4)

        self.fc1 = nn.Linear(in_features=in_features, out_features=no_neurons1)
        self.fc2 = nn.Linear(in_features=no_neurons1, out_features=no_neurons2)
        self.fc3 = nn.Linear(in_features=no_neurons2, out_features=no_neurons3)
        self.fc4 = nn.Linear(in_features=no_neurons3, out_features=no_neurons4)
        self.fc5 = nn.Linear(in_features=no_neurons4,
                             out_features=out_features)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.batch1(x)
        x = F.max_pool1d(x, 2)
        x = F.dropout(x, p=0.15)

        x = F.relu(self.conv2(x))
        x = self.batch2(x)
        x = F.max_pool1d(x, 2)
        x = F.dropout(x, p=0.15)

        x = F.relu(self.conv3(x))
        x = self.batch3(x)
        x = F.max_pool1d(x, 2)
        x = F.dropout(x, p=0.15)

        x = F.relu(self.conv4(x))
        x = self.batch4(x)
        x = F.max_pool1d(x, 2)
        x = F.dropout(x, p=0.15)

        # print(x.shape)

        x = x.view(-1, in_features)

        # print(x.shape)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)

        return F.log_softmax(x, dim=1)
