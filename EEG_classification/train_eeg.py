from VGG import VGG, Args
from matplotlib import pyplot
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import os
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

args = Args()

use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")


def train(args, model, device, train_loader, optimizer, epoch, weight):

    model.train()

    all_losses = []

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device).float(), target.to(device).long()

        optimizer.zero_grad()

        output = model(data)
        loss = F.nll_loss(output, target, weight)

        all_losses.append(loss.detach().cpu().numpy())

        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    return np.array(all_losses).mean()


def test(args, model, device, test_loader, weight):

    model.eval()
    test_loss = 0
    correct = 0
    conf = None

    with torch.no_grad():

        num_iter = 0
        for data, target in test_loader:

            data, target = data.to(device).float(), target.to(device).long()

            output = model(data)

            test_loss += F.nll_loss(output, target, weight)

            output = np.e ** output

            pred = output.argmax(dim=1, keepdim=True)

            conf = confusion_matrix(target.cpu(), pred.cpu())

            print('Predicted 1s : {}'.format(torch.sum(pred)))

            correct += pred.eq(target.view_as(pred)).float().mean().item()

            num_iter += 1

    print(conf)

    test_loss /= num_iter
    test_accuracy = 100. * correct / num_iter

    print('\nTest set: Average loss: {:.4f}, Accuracy: ({:.0f}%)\n'.format(
        test_loss, test_accuracy))

    return test_loss, test_accuracy


npz_file = np.load('final_signals.npz', allow_pickle=True)
signals = npz_file['signals']
labels = npz_file['labels']

perm = np.random.permutation(signals.shape[0])


no_train = int(0.7 * signals.shape[0])


dataset_signals, dataset_labels = map(torch.tensor, (signals, labels))

dataset_signals = dataset_signals.to(device)
dataset_labels = dataset_labels.to(device).long()


print('Signals shape is : {} and type is {}'.format(
    dataset_signals.shape, dataset_signals.dtype))

print('Labels shape is : {} and type is {}'.format(
    dataset_labels.shape, dataset_labels.dtype))


'''
creste weight_loss[0] daca vrei sa prezica mai mult 0

'''
weight_loss = torch.zeros(2)

weight_loss[0] = np.sum(labels) / labels.shape[0]

weight_loss[1] = (labels.shape[0] - np.sum(labels)) / labels.shape[0]


weight_loss = weight_loss.to(device).float()

print('Weight loss tensor : {}'.format(weight_loss))


train_dataset = TensorDataset(
    dataset_signals[perm[:no_train]], dataset_labels[perm[:no_train]])
test_dataset = TensorDataset(
    dataset_signals[perm[no_train:]], dataset_labels[perm[no_train:]])


def create_datasets():

    no_val = int(0.85 * signals.shape[0])

    np.savez("eeg-seizure_train.npz",
             train_signals=dataset_signals[perm[:no_train]], train_labels=dataset_labels[perm[:no_train]])

    np.savez("eeg-seizure_val.npz",
             val_signals=dataset_signals[perm[no_train:no_val]], val_labels=dataset_labels[perm[no_train:no_val]])

    np.savez("eeg-seizure_test.npz",
             test_signals=dataset_signals[perm[no_val:]])
    np.savez("eeg-seizure_test_labels.npz",
             test_labels=dataset_labels[perm[no_val:]])

    print('Seizures in train:')

    print(np.sum(np.array(dataset_labels[perm[:no_train]]), axis=0))
    print('of')
    print(no_train)

    print('Seizures in test and val:')
    print(np.sum(np.array(dataset_labels[perm[no_train:]]), axis=0))
    print('of')
    print(dataset_labels.shape[0] - no_train)


create_datasets()


train_loader = DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)


model = CNN().to(device)


optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

losses_train = []
losses_test = []
accuracy_test = []

for epoch in range(1, args.epochs + 1):

    train_loss = train(args, model, device, train_loader,
                       optimizer, epoch, weight_loss)

    test_loss, test_accuracy = test(
        args, model, device, test_loader, weight_loss)

    losses_train.append(train_loss)
    losses_test.append(test_loss)
    accuracy_test.append(test_accuracy)


def plot_loss(loss, label, color='blue'):
    pyplot.plot(loss, label=label, color=color)
    pyplot.legend()


pyplot.figure(1)
plot_loss(losses_train, 'train_loss', 'red')
plot_loss(losses_test, 'test_loss')
pyplot.savefig('losses.png')
pyplot.figure(2)
plot_loss(accuracy_test, 'test_accuracy')
pyplot.savefig('accuracy.png')

pyplot.show()

torch.save(model.state_dict(), "mnist_eeg.pt")


test = np.load('eeg-seizure_test_labels.npz', allow_pickle=True)
labels = test['test_labels']

print(labels.shape)
f = open('test.csv', 'w')

for i in range(0, labels.shape[0]):
    f.write(str(i) + ',' + str(labels[i]) + '\n')
