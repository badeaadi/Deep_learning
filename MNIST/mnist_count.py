from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from IPython.core.debugger import set_trace
import numpy as np
from matplotlib import pyplot



import pickle

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader



class Args():
  def __init__(self):
      self.batch_size = 256
      self.test_batch_size = 64
      self.epochs = 3
      self.lr = 0.01
      self.momentum = 0.9
      self.seed = 1
      self.log_interval = int(1000 / self.batch_size)
      self.cuda = False


kwargs={}

args = Args()
args.log_interval /= 10



def get_large_dataset(path, max_batch_idx=100, shuffle=False, first_k=5000):

  with open(path,'rb') as handle:
    data = pickle.load(handle)

  np_dataset_large  = np.expand_dims(data['images'],1)[:first_k]
  np_dataset_no_count = data['no_count'].astype(np.float32)[:first_k]
  
  print(f'np_dataset_large shape: {np_dataset_large.shape}')
  
  dataset_large, dataset_no_count= map(torch.tensor, 
                (np_dataset_large, np_dataset_no_count))


  large_dataset = TensorDataset(dataset_large, dataset_no_count)
  large_data_loader = DataLoader(large_dataset, batch_size=args.batch_size, shuffle=shuffle, drop_last=True)
  
  return large_data_loader

path_train = 'mnist_count_train.pickle'
path_test = 'mnist_count_test.pickle'





no_filters1 = 20
no_filters2 = 50
no_neurons1 = 1024
no_neurons2 = 64
no_neurons3 = 5

class CNN_count(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = no_filters1, kernel_size = 5, stride = 1)
        self.conv2 = nn.Conv2d(in_channels = no_filters1, out_channels = no_filters2, kernel_size = 5, stride = 1)
        
        self.fc1 = nn.Linear(in_features = 22 * 22 * no_filters2, out_features = no_neurons1)
        self.fc2 = nn.Linear(in_features = no_neurons1, out_features = no_neurons2)
        self.fc3 = nn.Linear(in_features = no_neurons2, out_features = 5)

    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        
        x = x.view(-1, 22 * 22 * no_filters2)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

x = torch.zeros(1, 1, 100, 100)
model = CNN_count()
print(model.forward(x).shape)



acuraccies = []
fractions = [0.2, 0.5, 1.0]

for k in fractions:
        
    round_k = int(5000 * k)
    large_data_loader_train = get_large_dataset(path_train, max_batch_idx=50,shuffle=True, first_k=round_k)
    large_data_loader_test = get_large_dataset(path_test, max_batch_idx=50) 
    
    
    # define two functions, one for training the model and one for testing it
    
    def train_count(args, model, train_loader, optimizer, epoch):
        
        model.train()
        all_losses = []
        batch_id = 0
        criterion = nn.CrossEntropyLoss()
    
        print("Train starting...")
        for data, target in train_loader:
            batch_id += 1
    
            data, target = data.float() / 255.0, target.long() - 1
            
            optimizer.zero_grad()
                
            output = model(data)
    
            loss = criterion(output, target)
    
            all_losses.append(loss.detach().cpu().numpy())
            loss.backward()
    
            optimizer.step()
            
            if batch_id % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_id * len(data), len(train_loader.dataset),
                    100. * batch_id / len(train_loader), loss.item()))
                
        return np.array(all_losses).mean()
    
    def test_count(args, model, test_loader):
    
        model.eval()
        test_loss = 0
        correct = 0
        
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            num_iter = 0
            for data, target in test_loader:
    
                data, target = data.float() / 255.0, target.long() - 1
                
                output = model(data).detach().cpu()
                
                test_loss += criterion(output, target)
    
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).float().mean().item()
    
                num_iter += 1
    
        test_loss /= num_iter
        test_accuracy = 100. * correct / num_iter
    
        print('\nTest set: Average loss: {:.4f}, Accuracy: ({:.0f}%)\n'.format(
            test_loss,
            test_accuracy))
        return test_loss, test_accuracy
    
    
    
    model = CNN_count()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    losses_train = []
    losses_test = []
    accuracy_test = []
    last_accuracy = 0
    
    for epoch in range(1, args.epochs + 1):
      
        train_loss = train_count(args, model, large_data_loader_train, optimizer, epoch)
    
        test_loss, test_accuracy = test_count(args, model, large_data_loader_test)
    
        losses_train.append(train_loss)
        losses_test.append(test_loss)
        accuracy_test.append(test_accuracy)
        last_accuracy = test_accuracy
    
    acuraccies.append(last_accuracy)
    
    def plot_loss(loss, label):
        pyplot.plot(loss, label=label)
        pyplot.legend()
    
    pyplot.figure(1)
    
    plot_loss(losses_train,'train_loss' + str(k))
    plot_loss(losses_test,'test_loss' + str(k))
    
    pyplot.figure(2)
    plot_loss(accuracy_test,'test_accuracy' + str(k))
    
    
    torch.save(model.state_dict(),"mnist_cnn_count.pt")
    

pl, ax = pyplot.subplots(1, 1)
ax.plot(fractions, acuraccies, 'r-', lw=3, alpha=0.6, label='Acuraccy with respect to data usage')



















