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
import numpy as np
from PIL import Image

import parameter
from libpy import Init
#import NLRWClass
import NLClass
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_flag = parameter.model_flag
device = parameter.device
batch_size = 500
test_batch_size = 2000
epochs = parameter.epochs
lr = 0.001           
momentum = parameter.momentum
seed = parameter.seed
flag_auto = parameter.flag_auto
log_interval = parameter.log_interval
patch_size  = 6
strick_size = 3
input_size  = 2000000
output_size = 300

if device == "cuda":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.manual_seed(seed)


# Classification Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.Classification = NLClass.GMMDense(input_features = patch_size * patch_size, output_features = output_size, device = "cuda")
        #self.Classification = NLClass.NLRWDense(input_features = patch_size * patch_size, output_features = output_size, work_style = "RW", UL_distant = 0.1, UU_distant = 0.5, device = "cuda")

    def forward(self, x):
        return self.Classification(x)


# Data Loader Class
class patch_data(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x_data = x
        self.y_data = y
        self.length = len(x)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.length


# M Step: Training Processing
def train( model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        YData = [[0 for n in range(output_size)] for n in range(len(target))]
        for i in range(0, len(target)):
            YData[i][target[i]] = 1
        data, YData = data.to(device), torch.Tensor(YData).to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.binary_cross_entropy(output, YData)

        loss.backward()
        optimizer.step()
        if not flag_auto:
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t\t\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.item()), end = "\r")
    if not flag_auto:
        print()


# E Step: Testing Processing
def test(model, device, test_loader, save_model, data_y):
    model.eval()
    test_loss = 0
    correct = 0
    results = torch.zeros([input_size], dtype = torch.long).to(device)
    
    loc = 0
    with torch.no_grad():
        for data, target in test_loader:
            YData  = [[0 for n in range(output_size)] for n in range(len(target))]
            for i in range(0, len(target)):
                YData[i][target[i]] = 1
            data, YData, target = data.to(device), torch.Tensor(YData).to(device), target.to(device)
            output = model(data)
            test_loss += F.binary_cross_entropy(output, YData, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            results[loc:loc+len(pred)] = torch.reshape(pred, [-1]).to(device)
            loc += test_batch_size
        if save_model:
            output = model(torch.rand([1, 1, patch_size * patch_size]).to(device))
    
    test_loss /= len(data_y)
    correct = results.eq(data_y.view_as(results)).sum().item() 
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} '.format(test_loss, correct, len(test_loader.dataset),))

    if correct / len(test_loader.dataset) > 0.9:
        print("HIGH ACCURACY, TERMINATED THE PROCESSING")
        with torch.no_grad():
            output = model(torch.rand([1, 1, patch_size * patch_size]).to(device))
        import os
        os._exit()
    
    return results


# Main Function
def main():
    #Read Images Location
    FileLoc = "./Input/patch_train"
    file_dir, files = Init.GetSufixFile(FileLoc, ["jpg"])

    #Patch Segmentation
    patchs = []
    for i in range(0, len(file_dir)):
        img = np.array(Image.open(file_dir[i]).convert("L"))
        p = 0
        q = 0
        stop_patch = False
        while 1:
            if len(patchs) != 0 and len(patchs) % 100000 == 0:
                print(len(patchs), "/", input_size, end = "\r")
            if q + patch_size + strick_size >= len(img[i]):
                p += strick_size
                q = 0
            if p + patch_size >= len(img):
                break
            q += strick_size
            sub_patch = img[p: p + patch_size, q: q + patch_size]
            sub_patch = sub_patch.reshape([-1])
            patchs.append(torch.Tensor(sub_patch).to(device))
            if len(patchs) >= input_size:
                stop_patch = True
                break
        if stop_patch:
            break

    print("Total patchs:", len(patchs), "Patch size: ", patchs[0].size(), "Output Class", output_size)
    
    loss_rate = lr
    #Initial Model
    model     = Net().to(device)
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=loss_rate, momentum=momentum)
    
    #Initial Data
    data_x      = patchs
    data_y      = torch.LongTensor([0 for n in range(input_size)]).to(device)
    data_set    = patch_data(data_x, data_y)


    #Main Cluster Loop
    for epoch in range(1, epochs + 1):
        print("Looping epoch = ", epoch)
        #E Step
        start        = time.time()
        test_loader  = torch.utils.data.DataLoader(data_set, batch_size = test_batch_size, shuffle = False)
        if epoch % 5 == 0 and epoch != 0:
            data_y   = test(model, device, test_loader, True, data_y)
            loss_rate /= 10
            optimizer = optim.SGD(model.parameters(), lr=loss_rate, momentum=momentum)
        else:
            data_y   = test(model, device, test_loader, False, data_y)
        end          = time.time()
        t2           = end - start

        #Data Reflesh
        data_set     = patch_data(data_x, data_y)

        #M Step
        start        = time.time()
        train_loader = torch.utils.data.DataLoader(data_set, batch_size = batch_size, shuffle = True)
        train(model, device, train_loader, optimizer, epoch)
        end          = time.time()
        t1           = end - start
        
        print("Time Usage: E Step:", t2, "M Step", t1)


    
if __name__ == '__main__':
    main()
