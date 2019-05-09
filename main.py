#=============================================================================
#
#       Group Attribute Random Walk Program
#       Main.py
#
#       Copyright by KazukiAmakawa, all right reserved
#
#=======================================================
#
#       Intro
#       This is the Main file we build in MNIST recoginization.
#
#=============================================================================

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time

from libpy import Init
import NLRWClass

flag_auto = False
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv2d1 = nn.Conv2d(in_channels = 1,  out_channels = 7,  kernel_size = 3, stride = 1, padding = 1)
        self.conv2d2 = nn.Conv2d(in_channels = 7,  out_channels = 14, kernel_size = 3, stride = 1, padding = 1)
        self.conv2d3 = nn.Conv2d(in_channels = 14, out_channels = 28, kernel_size = 3, stride = 1, padding = 1)
        
        self.fc1 = nn.Linear(252, 64)

        #self.Classification = NLRWClass.NLRWDense(input_features = 64, output_features = 10, work_style = "NL", UL_distant = 1, UU_distant = 1, device = "cuda")
        self.Classification = NLRWClass.NLRWDense(input_features = 64, output_features = 10, work_style = "RW", UL_distant = 0.1, UU_distant = 0.1, device = "cuda")
        #self.Classification = nn.Linear(64, 10, bias = False)
        

    def forward(self, x):
        Block1 = self.conv2d1(x)
        Block1 = F.leaky_relu(Block1, 0.1)
        Block1 = F.max_pool2d(Block1, 2)

        Block2 = self.conv2d2(Block1)
        Block2 = F.leaky_relu(Block2, 0.1)
        Block2 = F.max_pool2d(Block2, 2)

        Block3 = self.conv2d3(Block2)
        Block3 = F.leaky_relu(Block3, 0.1)
        Block3 = F.max_pool2d(Block3, 2)
        #Finals = Block3.view(1)        # To get the vector length

        Finals = Block3.view(-1, 252)
        Finals = self.fc1(Finals)
        Finals = F.leaky_relu(Finals, 0.1)

        Finals = self.Classification(Finals)
        #Finals = F.softmax(Finals)
        return Finals



def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        YData = [[0 for n in range(10)] for n in range(len(target))]
        for i in range(0, len(target)):
            YData[i][target[i]] = 1
        data, YData = data.to(device), torch.Tensor(YData).to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.binary_cross_entropy(output, YData)

        loss.backward()
        optimizer.step()
        if not flag_auto:
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t\t\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.item()), end = "\r")
    if not flag_auto:
        print()

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            YData = [[0 for n in range(10)] for n in range(len(target))]
            for i in range(0, len(target)):
                YData[i][target[i]] = 1
            data, YData, target = data.to(device), torch.Tensor(YData).to(device), target.to(device)
            output = model(data)
            test_loss += F.binary_cross_entropy(output, YData, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset) * 10

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} '.format(test_loss, correct, len(test_loader.dataset),))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1)')
    parser.add_argument('--epochs', type=int, default=150, metavar='N',
                        help='number of epochs to train (default: 150)')
    parser.add_argument('--lr', type=float, default=0.2, metavar='LR',
                        help='learning rate (default: 0.02), Warning, if you using random walk with parameter, it is necessary to change this loss rate with parameter')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    
    if use_cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.multiprocessing.set_start_method("spawn")
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('Input/data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('Input/data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = Net().to(device)
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        print("Looping epoch = ", epoch)
        start = time.time()
        train(args, model, device, train_loader, optimizer, epoch)
        end = time.time()
        t1 = end - start

        #test(args, model, device, train_loader)
        start = time.time()
        test(args, model, device, test_loader)
        end = time.time()
        t2 = end - start
        print("Time Usage: Training time", t1, "Testing time", t2)

    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")
        
if __name__ == '__main__':
    main()
