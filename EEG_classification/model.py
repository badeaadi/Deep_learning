import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from matplotlib import pyplot

in_channels = 23
no_filters1 = 46
no_filters2 = 92
no_neurons1 = 2000

in_features = 46 * 250

class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels = in_channels, out_channels = no_filters1, kernel_size = 7, stride = 1)
        self.conv2 = nn.Conv1d(in_channels = no_filters1, out_channels = no_filters2, kernel_size = 4, stride = 2)
        
        self.fc1 = nn.Linear(in_features = in_features, out_features = no_neurons1)
        self.fc2 = nn.Linear(in_features = no_neurons1, out_features = 2)
     
    def forward(self, x):
      
        x = F.relu(self.conv1(x))
    
        x = x.view(-1, in_features)
                
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
            
        return F.log_softmax(x, dim = 1)


def train(args, model, device, train_loader, optimizer, epoch, weight_loss):
    
    model.train()

    all_losses = []

    
    for batch_idx, (data, target) in enumerate(train_loader):
    
        data, target = data.to(device).float(), target.to(device).long()

        optimizer.zero_grad()
        
        output = model(data)
        loss = F.nll_loss(output, target, weight = weight_loss)
        
        all_losses.append(loss.detach().cpu().numpy())
        
        loss.backward()
        optimizer.step()
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
    return np.array(all_losses).mean()

def test(args, model, device, test_loader):
    
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        
        num_iter = 0
        for data, target in test_loader:
            
            data, target = data.to(device).float(), target.to(device).long()
            output = model(data)
            
            test_loss += F.nll_loss(output, target)

            pred = output.argmax(dim=1, keepdim=True)
            
            print('Predicted 1s : {}'.format(torch.sum(pred)))
            
            correct += pred.eq(target.view_as(pred)).float().mean().item()
            
            num_iter += 1

    test_loss /= num_iter
    test_accuracy = 100. * correct / num_iter

    print('\nTest set: Average loss: {:.4f}, Accuracy: ({:.0f}%)\n'.format(test_loss, test_accuracy))
    return test_loss, test_accuracy




kwargs={}

class Args():
  def __init__(self):
      self.batch_size = 256
      self.test_batch_size = 64
      self.epochs = 100
      self.lr = 0.05
      self.momentum = 0.9
      self.seed = 1
      self.log_interval = int(1000 / self.batch_size)
      self.cuda = True

args = Args()

use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}




npz_file = np.load('signals0.npz', allow_pickle = True)
signals = npz_file['signals']
labels = npz_file['labels']

np.random.shuffle(signals)
np.random.shuffle(labels)
no_train = int(0.8 * signals.shape[0])
    
print(signals.shape)
print(labels.shape)

dataset_signals, dataset_labels = map(torch.tensor, (signals, labels))

dataset_signals = dataset_signals.to(device)
dataset_labels = dataset_labels.to(device)



weight_loss = torch.zeros(2, dtype = float).float()

weight_loss[0] = np.sum(labels) / labels.shape[0]
weight_loss[1] = (labels.shape[0] - np.sum(labels)) / labels.shape[0]
print(weight_loss)


train_dataset = TensorDataset(dataset_signals[:no_train], dataset_labels[:no_train])
test_dataset = TensorDataset(dataset_signals[no_train:], dataset_labels[no_train:])

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True )


model= CNN().to(device).float()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

losses_train = []
losses_test = []
accuracy_test = []

for epoch in range(1, args.epochs + 1):
  
    train_loss = train(args, model, device, train_loader, optimizer, epoch, weight_loss)
    test_loss, test_accuracy = test(args, model, device, test_loader)

    losses_train.append(train_loss)
    losses_test.append(test_loss)
    accuracy_test.append(test_accuracy)

def plot_loss(loss, label, color='blue'):
    pyplot.plot(loss, label=label, color=color)
    pyplot.legend()

pyplot.figure(1)
plot_loss(losses_train,'train_loss','red')
plot_loss(losses_test,'test_loss')
pyplot.figure(2)
plot_loss(accuracy_test,'test_accuracy')

torch.save(model.state_dict(),"mnist_eeg.pt")
