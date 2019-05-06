#=============================================================================
#
#       Group Attribute Random Walk Program
#       garw.py
#
#       Copyright by KazukiAmakawa, all right reserved
#
#=======================================================
#
#       Intro
#       This is the main project we build for pedestrian re-identification
#       with group attribute random walk model
#
#=============================================================================
from __future__ import print_function
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import *

from libpy import Init
import NLRWClass
import CUB_load
import parameter

#Initialization======================================================

#Read parameter from parameter.py file
model_flag = parameter.model_flag
device = parameter.device
batch_size = parameter.batch_size
test_batch_size = parameter.test_batch_size
epochs = parameter.epochs
lr = parameter.lr           
momentum = parameter.momentum
seed = parameter.seed
log_interval = parameter.log_interval
save_model = parameter.save_model
flag_auto = parameter.flag_auto
map_size = parameter.map_size
featurea_length = parameter.featurea_length


pretrain_model = eval(model_flag)(pretrained=True)
attr_class = sum(nb_attributes)
total_class = len(nb_attributes)

if device == "cuda":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.multiprocessing.set_start_method("spawn")
torch.manual_seed(args.seed)




#Definition Classification============================================
class attribute_net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.map2attr = NLRWClass.NLRWDense(input_features = map_size, output_features = attr_class, work_style = "RW", UL_distant = 0.1, UU_distant = 0.1, device = device)
        #self.attr2final = NLRWClass.NLRWDense(input_features = attr_class, output_features = total_class, work_style = "RW", UL_distant = 0.1, UU_distant = 0.1, device = device)
        #self.map2final = NLRWClass.NLRWDense(input_features = featurea_length, output_features = total_class, work_style = "RW", UL_distant = 0.1, UU_distant = 0.1, device = device)
        self.attr2final = nn.Linear(attr_class, total_class, bias = False)
        self.map2final = nn.Linear(featurea_length, total_class, bias = False)

    def forward(self, x):
        #Input [[None, [512, 196]], [None, 512]]
        attr_map, features = x
        attr_dis = torch.Tensor().to(device)
        for i in range(0, len(attr_map)):
            attr_pdf = self.map2attr(attr_map[i])
            attr_sum = torch.sum(attr_pdf.t(), dim = 1).t()
            attr_dis = torch.cat(attr_dis, attr_sum)

        attr_dis = attr_dis.reshape(-1, attr_class)

        attr_final = self.attr2final(attr_dis)
        attr_final = F.softmax(attr_final)

        map_final = self.map2final(features)
        map_final = F.softmax(map_final)
        
        final = (attr_final + map_final) / 2
        
        return (attr_dis, final)





#Main Training and Testing ===========================================
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target, attr) in enumerate(train_loader):
        data, target, attr = data.to(device), torch.Tensor(YData).to(device)
        optimizer.zero_grad()
        attr_dis, final = model(data)
        loss = F.binary_cross_entropy(output, YData)

        loss.backward()
        optimizer.step()
        if not flag_auto:
            if batch_idx % log_interval == 0:
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




#Main=================================================================
def main():
    #Input Images, attribute and final target for input data
    input_data, input_attr, input_target = load_data("./CUB_200_2011")

    feature_map, final_feature = sharedCNN(input_data)

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

