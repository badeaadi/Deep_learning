import csv
from CNN import CNN, Args
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

    print('\nValidation set: Average loss: {:.4f}, Accuracy: ({:.0f}%)\n'.format(
        test_loss, test_accuracy))

    return test_loss, test_accuracy


def evaluate(args, model, device, data, weight):
    with torch.no_grad():
        data = torch.tensor(data)
        data = data.to(device).float()

        output = model(data)

        output = np.e ** output

        pred = output.argmax(dim=1, keepdim=True)

        print('Predicted 1s : {}'.format(torch.sum(pred)))

    return pred


training_file = np.load('eeg-seizure_train.npz', allow_pickle=True)
train_signals = training_file['train_signals']
train_labels = training_file['train_labels']

val_file = np.load('eeg-seizure_val.npz', allow_pickle=True)
val_signals = val_file['val_signals']
val_labels = val_file['val_labels']

test_file = np.load('eeg-seizure_test.npz', allow_pickle=True)
test_signals = test_file['test_signals']

scaler = StandardScaler()
scaler.fit(train_signals.reshape((-1, 23 * 256)))

train_signals = scaler.transform(
    train_signals.reshape((-1, 23 * 256))).reshape(train_signals.shape)
val_signals = scaler.transform(
    val_signals.reshape((-1, 23 * 256))).reshape(val_signals.shape)
test_signals = scaler.transform(
    test_signals.reshape((-1, 23 * 256))).reshape(test_signals.shape)

train_dataset_signals, train_dataset_labels = map(
    torch.tensor, (train_signals, train_labels))
val_dataset_signals, val_dataset_labels = map(
    torch.tensor, (val_signals, val_labels))

'''
creste weight_loss[0] daca vrei sa prezica mai mult 0

'''
weight_loss = torch.zeros(2)

weight_loss[0] = np.sum(train_labels) / train_labels.shape[0]

weight_loss[1] = (train_labels.shape[0] -
                  np.sum(train_labels)) / train_labels.shape[0]


weight_loss = weight_loss.to(device).float()

print('Weight loss tensor : {}'.format(weight_loss))


train_dataset = TensorDataset(
    train_dataset_signals, train_dataset_labels)
val_dataset = TensorDataset(
    val_dataset_signals, val_dataset_labels)


train_loader = DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(
    val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)


model = CNN().to(device)


optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
# optimizer = optim.Adam(model.parameters(), lr=args.lr, momentum=args.momentum)

losses_train = []
losses_test = []
accuracy_test = []

for epoch in range(1, args.epochs + 1):

    train_loss = train(args, model, device, train_loader,
                       optimizer, epoch, weight_loss)

    test_loss, test_accuracy = test(
        args, model, device, val_loader, weight_loss)

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

model.load_state_dict(torch.load("mnist_eeg.pt"))
predictions = evaluate(args, model, device, test_signals,
                       weight_loss).cpu().detach().numpy()

with open("predictions.csv", mode="w") as csv_file:
    writer = csv.writer(csv_file, delimiter=',', lineterminator='\n')

    writer.writerow(['ID', 'PREDICTED'])

    for i in range(predictions.shape[0]):
        writer.writerow([i, predictions[i][0]])
