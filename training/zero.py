# Train different models on different standard datasets 
# and compare their performance

import torchvision.models as models
import torch.nn as nn
import torchvision
import torch 
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import argparse
import torch.optim as optim

from dataloaders import get_dataloaders

def train(model, args, device, train_loader, optimizer, epoch):
    model.train()
    # log the performances every epoch
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return train_loss

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return test_loss, (100. * correct / len(test_loader.dataset))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')    
    parser.add_argument('--log-dir', type=str, default='/nfs/users/ext_shikhar.srivastava/workspace/TANS/training/logs/',)

    parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='N',)
    parser.add_argument('--model', type=str, default='resnet18', metavar='N',)

    parser.add_argument('--num-workers', type=int, default=0, metavar='N',)

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader, test_loader, trainset, _ = get_dataloaders(args, **kwargs)


    model = getattr(models, args.model)(num_classes = len(trainset.classes)).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # log metrics
    metrics = {'epoch':[],'train_loss': [], 'test_loss': [], 'test_acc': []}
    for epoch in range(1, args.epochs + 1):
        tr_loss = train(model, args, device, train_loader, optimizer, epoch)
        te_loss, acc =  test(model, device, test_loader)
        metrics['train_loss'].append(tr_loss)
        metrics['test_loss'].append(te_loss)
        metrics['test_acc'].append(acc)
        metrics['epoch'].append(epoch)
        torch.save(metrics, args.log_dir + args.dataset + '_' + args.model + '_metrics.pt')

    if (args.save_model):
        torch.save(model, args.log_dir + args.dataset + '_' + args.model + '.pt')
        torch.save(metrics, args.log_dir + args.dataset + '_' + args.model + '_metrics.pt')

if __name__ == '__main__':
    main()

# python main.py --save-model --no-cuda --batch-size=128 --test-batch-size=1000 --epochs=10 --lr=0.001 --momentum=0.9 --log-interval=10 --seed=1
