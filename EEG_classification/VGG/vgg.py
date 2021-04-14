import torch.nn.functional as F
import torch.nn as nn
import torch


class Args():
    def __init__(self):

        self.batch_size = 512
        self.epochs = 300
        self.lr = 1e-5
        self.final_lr = 0.1
        self.momentum = 0.9
        self.seed = 42
        self.log_interval = int(8000 / self.batch_size)
        self.cuda = True
        self.weight_decay = 1e-6


class VGG(nn.Module):
    def __init__(self):
        super().__init__()

        # 1 channel input, since we are using grayscale images
        self.layer1_1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.layer1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.layer1_mp = nn.MaxPool2d(2, 2)

        self.layer2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.layer2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.layer2_mp = nn.MaxPool2d(2, 2)

        self.layer3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.layer3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.layer3_3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.layer3_mp = nn.MaxPool2d(2, 2)

        self.layer4_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.layer4_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.layer4_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.layer4_mp = nn.MaxPool2d(2, 2)

        self.layer5_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.layer5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.layer5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.layer5_mp = nn.MaxPool2d(2, 2)

        # conv layers done, time for linear
        self.fc1 = nn.Linear(512 * 1 * 1, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 2)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)

        x = F.relu(self.layer1_1(x))
        x = F.relu(self.layer1_2(x))
        x = self.layer1_mp(x)

        x = F.relu(self.layer2_1(x))
        x = F.relu(self.layer2_2(x))
        x = self.layer2_mp(x)

        x = F.relu(self.layer3_1(x))
        x = F.relu(self.layer3_2(x))
        x = F.relu(self.layer3_3(x))
        x = self.layer3_mp(x)

        x = F.relu(self.layer4_1(x))
        x = F.relu(self.layer4_2(x))
        x = F.relu(self.layer4_3(x))
        x = self.layer4_mp(x)

        x = F.relu(self.layer5_1(x))
        x = F.relu(self.layer5_2(x))
        x = F.relu(self.layer5_3(x))
        # x = self.layer5_mp(x)

        # we need to flatten x for the linear layers
        x = x.view(-1, 512 * 1 * 1)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.25)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.25)
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)
