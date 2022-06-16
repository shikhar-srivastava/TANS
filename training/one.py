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

        loss = F.cross_entropy(output, target, reduction='sum')
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    train_loss /= len(train_loader.dataset)
    return train_loss

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return test_loss, correct, len(test_loader.dataset), (100. * correct / len(test_loader.dataset))

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

    parser.add_argument('--dataset', type=str, default='MNIST', metavar='N',)
    parser.add_argument('--model', type=str, default='resnet18', metavar='N',)
    parser.add_argument('--model-path', default='/nfs/users/ext_shikhar.srivastava/workspace/TANS/training/logs/MNIST/shufflenet_v2_x0_5MNIST_shufflenet_v2_x0_5.pt', \
        type=str)
    parser.add_argument('--num-workers', type=int, default=0, metavar='N',)

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader, test_loader, trainset, _ = get_dataloaders(args, **kwargs)
    
    if args.dataset == 'USPS':
        num_classes = len(set(trainset.targets))
    elif args.dataset == "SVHN":
        num_classes = len(set(trainset.labels))
    else:
        num_classes = len(trainset.classes)

    # Load model and remove the last layer
    #model = getattr(models, args.model)(num_classes = num_classes).to(device)
    
    model = torch.load(args.model_path)

    print(model.__class__.__name__ )
    if model.__class__.__name__ == 'ResNet':
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        model.fc.weight.data.zero_()
        model.fc.bias.data.zero_()
    
    elif ((model.__class__.__name__ == 'MobileNetV2') | (model.__class__.__name__ == 'EfficientNet') \
        | (model.__class__.__name__ == 'MobileNetV3') | (model.__class__.__name__ == 'VGG')  ):
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
        model.classifier[-1].weight.data.zero_()
        model.classifier[-1].bias.data.zero_()
    elif model.__class__.__name__ == 'SqueezeNet':
        model.classifier.Conv2d = torch.nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model.classifier.Conv2d.weight.data.zero_()
        model.classifier.Conv2d.bias.data.zero_()

    elif model.__class__.__name__ == 'ShuffleNetV2':
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        model.fc.weight.data.zero_()
        model.fc.bias.data.zero_()
    else:
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
        model.classifier.weight.data.zero_()
        model.classifier.bias.data.zero_()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # log metrics
    metrics = {'pre_train':[], 'epoch':[],'train_loss': [], 'test_loss': [], 'test_acc': [], 'test_correct': [], 'test_total': []}
    te_loss, test_correct, test_total, acc =  test(model, device, test_loader)
    metrics['pre_train'].append((te_loss,test_correct, test_total, acc))
    
    model_name = args.model_path.split('/')[-1].split('.')[0]
    log_path = args.log_dir + model_name + '_' + args.dataset

    for epoch in range(1, args.epochs + 1):
        tr_loss = train(model, args, device, train_loader, optimizer, epoch)
        te_loss, test_correct, test_total, acc =  test(model, device, test_loader)
        metrics['train_loss'].append(tr_loss)
        metrics['test_loss'].append(te_loss)
        metrics['test_acc'].append(acc)
        metrics['test_correct'].append(test_correct)
        metrics['test_total'].append(test_total)
        metrics['epoch'].append(epoch)
        torch.save(metrics, log_path + '_metrics.pt')

    if (args.save_model):
        torch.save(model, log_path + '.pt')
        torch.save(metrics, log_path + '_metrics.pt')

if __name__ == '__main__':
    main()

# python main.py --save-model --no-cuda --batch-size=128 --test-batch-size=1000 --epochs=10 --lr=0.001 --momentum=0.9 --log-interval=10 --seed=1
