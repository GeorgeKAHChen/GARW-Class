#=============================================================================
#
#       Group Attribute Random Walk Program
#       apr.py
#
#       Copyright by KazukiAmakawa, all right reserved
#
#=======================================================
#
#       Intro
#       This file rebuilded the APR(Attribute-Preson Recognization) with RW development
#
#=============================================================================

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import *
import numpy as np
import time
import os

from libpy import Init
import NLRWClass
import CUB_load
import parameter

#Initialization======================================================

#Read parameter from parameter.py file
model_flag = parameter.model_flag
device = parameter.device
batch_size = parameter.batch_size
share_batch_size = parameter.share_batch_size
test_batch_size = parameter.test_batch_size
epochs = parameter.epochs
lr = parameter.lr           
momentum = parameter.momentum
seed = parameter.seed
log_interval = parameter.log_interval
save_model = parameter.save_model
map_size = parameter.map_size
featurea_length = parameter.featurea_length
dataset_location = parameter.dataset_location
nb_attributes = parameter.nb_attributes
total_class = parameter.total_class
flag_auto = parameter.flag_auto
flag_all = parameter.flag_all
Lambda = 8

#Initial
attr_class = sum(nb_attributes) + len(nb_attributes)
attr_total = len(nb_attributes)

if device == "cuda":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    #torch.multiprocessing.set_start_method("spawn")
torch.manual_seed(seed)





#Definition Classification============================================
class attribute_net(nn.Module):
    def __init__(self):
        super(attribute_net, self).__init__()
        self.fca00l1 = nn.Linear(512 * parameter.map_size, 1000)
        self.fca00l2 = nn.Linear(1000, max(nb_attributes))
        self.fca01l1 = nn.Linear(512 * parameter.map_size, 1000)
        self.fca01l2 = nn.Linear(1000, max(nb_attributes))
        self.fca02l1 = nn.Linear(512 * parameter.map_size, 1000)
        self.fca02l2 = nn.Linear(1000, max(nb_attributes))
        self.fca03l1 = nn.Linear(512 * parameter.map_size, 1000)
        self.fca03l2 = nn.Linear(1000, max(nb_attributes))
        self.fca04l1 = nn.Linear(512 * parameter.map_size, 1000)
        self.fca04l2 = nn.Linear(1000, max(nb_attributes))
        self.fca05l1 = nn.Linear(512 * parameter.map_size, 1000)
        self.fca05l2 = nn.Linear(1000, max(nb_attributes))
        self.fca06l1 = nn.Linear(512 * parameter.map_size, 1000)
        self.fca06l2 = nn.Linear(1000, max(nb_attributes))
        self.fca07l1 = nn.Linear(512 * parameter.map_size, 1000)
        self.fca07l2 = nn.Linear(1000, max(nb_attributes))
        self.fca08l1 = nn.Linear(512 * parameter.map_size, 1000)
        self.fca08l2 = nn.Linear(1000, max(nb_attributes))
        self.fca09l1 = nn.Linear(512 * parameter.map_size, 1000)
        self.fca09l2 = nn.Linear(1000, max(nb_attributes))
        self.fca10l1 = nn.Linear(512 * parameter.map_size, 1000)
        self.fca10l2 = nn.Linear(1000, max(nb_attributes))
        self.fca11l1 = nn.Linear(512 * parameter.map_size, 1000)
        self.fca11l2 = nn.Linear(1000, max(nb_attributes))
        self.fca12l1 = nn.Linear(512 * parameter.map_size, 1000)
        self.fca12l2 = nn.Linear(1000, max(nb_attributes))
        self.fca13l1 = nn.Linear(512 * parameter.map_size, 1000)
        self.fca13l2 = nn.Linear(1000, max(nb_attributes))
        self.fca14l1 = nn.Linear(512 * parameter.map_size, 1000)
        self.fca14l2 = nn.Linear(1000, max(nb_attributes))
        self.fca15l1 = nn.Linear(512 * parameter.map_size, 1000)
        self.fca15l2 = nn.Linear(1000, max(nb_attributes))
        self.fca16l1 = nn.Linear(512 * parameter.map_size, 1000)
        self.fca16l2 = nn.Linear(1000, max(nb_attributes))
        self.fca17l1 = nn.Linear(512 * parameter.map_size, 1000)
        self.fca17l2 = nn.Linear(1000, max(nb_attributes))
        self.fca18l1 = nn.Linear(512 * parameter.map_size, 1000)
        self.fca18l2 = nn.Linear(1000, max(nb_attributes))
        self.fca19l1 = nn.Linear(512 * parameter.map_size, 1000)
        self.fca19l2 = nn.Linear(1000, max(nb_attributes))
        self.fca20l1 = nn.Linear(512 * parameter.map_size, 1000)
        self.fca20l2 = nn.Linear(1000, max(nb_attributes))
        self.fca21l1 = nn.Linear(512 * parameter.map_size, 1000)
        self.fca21l2 = nn.Linear(1000, max(nb_attributes))
        self.fca22l1 = nn.Linear(512 * parameter.map_size, 1000)
        self.fca22l2 = nn.Linear(1000, max(nb_attributes))
        self.fca23l1 = nn.Linear(512 * parameter.map_size, 1000)
        self.fca23l2 = nn.Linear(1000, max(nb_attributes))
        self.fca24l1 = nn.Linear(512 * parameter.map_size, 1000)
        self.fca24l2 = nn.Linear(1000, max(nb_attributes))
        self.fca25l1 = nn.Linear(512 * parameter.map_size, 1000)
        self.fca25l2 = nn.Linear(1000, max(nb_attributes))
        self.fca26l1 = nn.Linear(512 * parameter.map_size, 1000)
        self.fca26l2 = nn.Linear(1000, max(nb_attributes))
        self.fca27l1 = nn.Linear(512 * parameter.map_size, 1000)
        self.fca27l2 = nn.Linear(1000, max(nb_attributes))
        self.fcil1 = nn.Linear(512 * parameter.map_size, 1000)
        self.fcil2 = nn.Linear(1000, total_class)


    def forward(self, x):
        #Input [[None, [512, 49]], [None, 512]]
        attr_map, features = x
        attr_map = attr_map.view(-1, 512 * parameter.map_size)
        attr_prob = []
        attr = self.fca00l1(attr_map)
        attr = self.fca00l2(attr)
        attr_prob.append(F.softmax(attr, dim = 1))
        attr = self.fca01l1(attr_map)
        attr = self.fca01l2(attr)
        attr_prob.append(F.softmax(attr, dim = 1))
        attr = self.fca02l1(attr_map)
        attr = self.fca02l2(attr)
        attr_prob.append(F.softmax(attr, dim = 1))
        attr = self.fca03l1(attr_map)
        attr = self.fca03l2(attr)
        attr_prob.append(F.softmax(attr, dim = 1))
        attr = self.fca04l1(attr_map)
        attr = self.fca04l2(attr)
        attr_prob.append(F.softmax(attr, dim = 1))
        attr = self.fca05l1(attr_map)
        attr = self.fca05l2(attr)
        attr_prob.append(F.softmax(attr, dim = 1))
        attr = self.fca06l1(attr_map)
        attr = self.fca06l2(attr)
        attr_prob.append(F.softmax(attr, dim = 1))
        attr = self.fca07l1(attr_map)
        attr = self.fca07l2(attr)
        attr_prob.append(F.softmax(attr, dim = 1))
        attr = self.fca08l1(attr_map)
        attr = self.fca08l2(attr)
        attr_prob.append(F.softmax(attr, dim = 1))
        attr = self.fca09l1(attr_map)
        attr = self.fca09l2(attr)
        attr_prob.append(F.softmax(attr, dim = 1))
        attr = self.fca10l1(attr_map)
        attr = self.fca10l2(attr)
        attr_prob.append(F.softmax(attr, dim = 1))
        attr = self.fca11l1(attr_map)
        attr = self.fca11l2(attr)
        attr_prob.append(F.softmax(attr, dim = 1))
        attr = self.fca12l1(attr_map)
        attr = self.fca12l2(attr)
        attr_prob.append(F.softmax(attr, dim = 1))
        attr = self.fca13l1(attr_map)
        attr = self.fca13l2(attr)
        attr_prob.append(F.softmax(attr, dim = 1))
        attr = self.fca14l1(attr_map)
        attr = self.fca14l2(attr)
        attr_prob.append(F.softmax(attr, dim = 1))
        attr = self.fca15l1(attr_map)
        attr = self.fca15l2(attr)
        attr_prob.append(F.softmax(attr, dim = 1))
        attr = self.fca16l1(attr_map)
        attr = self.fca16l2(attr)
        attr_prob.append(F.softmax(attr, dim = 1))
        attr = self.fca17l1(attr_map)
        attr = self.fca17l2(attr)
        attr_prob.append(F.softmax(attr, dim = 1))
        attr = self.fca18l1(attr_map)
        attr = self.fca18l2(attr)
        attr_prob.append(F.softmax(attr, dim = 1))
        attr = self.fca19l1(attr_map)
        attr = self.fca19l2(attr)
        attr_prob.append(F.softmax(attr, dim = 1))
        attr = self.fca20l1(attr_map)
        attr = self.fca20l2(attr)
        attr_prob.append(F.softmax(attr, dim = 1))
        attr = self.fca21l1(attr_map)
        attr = self.fca21l2(attr)
        attr_prob.append(F.softmax(attr, dim = 1))
        attr = self.fca22l1(attr_map)
        attr = self.fca22l2(attr)
        attr_prob.append(F.softmax(attr, dim = 1))
        attr = self.fca23l1(attr_map)
        attr = self.fca23l2(attr)
        attr_prob.append(F.softmax(attr, dim = 1))
        attr = self.fca24l1(attr_map)
        attr = self.fca24l2(attr)
        attr_prob.append(F.softmax(attr, dim = 1))
        attr = self.fca25l1(attr_map)
        attr = self.fca25l2(attr)
        attr_prob.append(F.softmax(attr, dim = 1))
        attr = self.fca26l1(attr_map)
        attr = self.fca26l2(attr)
        attr_prob.append(F.softmax(attr, dim = 1))
        attr = self.fca27l1(attr_map)
        attr = self.fca27l2(attr)
        attr_prob.append(F.softmax(attr, dim = 1))
        identity = self.fcil1(attr_map)
        identity = self.fcil2(identity)
        id_prob = F.softmax(identity, dim = 1)
        return id_prob, attr_prob


class map_fea_tar_attr(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x_data = x
        self.y_data = y
        self.length = len(x)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.length
        




#Main Training and Testing ===========================================
def train(model, train_loader, optimizer, epoch):
    model.train()

    for batch_idx, (data, outputs) in enumerate(train_loader):
        
        target, attr = outputs
        
        #print(len(attr), len(attr[0]), attr[0][0].size())
        """
        target, attr_data = outputs
        
        attr = [[0 for n in range(len(attr_data[0]))] for n in range(len(attr_data))]
        for i in range(0, len(attr_data)):
            for j in range(0, len(attr_data[i])):
                attr[i][j] = attr_data[j][i]
        
        """
        target = torch.Tensor(target).to(device)
        #print(len(attr), attr[0].size())
        optimizer.zero_grad()
        id_prob, attr_prob = model(data)
        #print(id_prob.size())
        loss1 = F.binary_cross_entropy(id_prob, target)
        loss2 = 0
        for i in range(0, len(nb_attributes)):
            #print(attr_prob[i].size())
            #print(len(attr[i]))
            loss2 += F.binary_cross_entropy(attr_prob[i], attr[i])
        loss = Lambda * loss1 + loss2 / len(nb_attributes)
        loss.backward()
        optimizer.step()
        if not flag_auto:
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t\t\tLoss: {:.6f}'.format(epoch, batch_idx * batch_size, len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.item()), end = "\r")
    
    if not flag_auto:
        print()



def test(model, test_loader):
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, outputs in test_loader:
            target, attr = outputs
            
            target = torch.Tensor(target).to(device)
            id_prob, attr_prob = model(data)
        
            loss1 = F.binary_cross_entropy(id_prob, target)
            loss2 = 0
            for i in range(0, len(nb_attributes)):
                loss2 += F.binary_cross_entropy(attr_prob[i], attr[i])
            loss = Lambda * loss1 + loss2 / len(nb_attributes)
        
            pred = id_prob.argmax(dim=1, keepdim=True)
            goal = target.argmax(dim=1, keepdim=True)
            correct += pred.eq(goal.view_as(pred)).sum().item()

    loss /= len(test_loader.dataset) * test_batch_size

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} '.format(loss, correct, len(test_loader.dataset),))



def build_shared_CNN():
    #I still need to write a brach calculation
    shared_model = eval(model_flag)(pretrained=True)
    shared_model = shared_model.to("cpu")

    #Build map part of sharedCNN
    map_model = nn.Sequential(*list(shared_model.children())[:-1])
    map_model = nn.DataParallel(map_model,device_ids=[0,1])
    map_model.to(device)
    #map_model = map_model.to("cpu")

    #Build feature part of sharedCNN
    feature_model = nn.Sequential(*list(shared_model.children())[-1])
    feature_model = nn.DataParallel(feature_model,device_ids=[0,1])
    feature_model.to(device)
    #feature_model = feature_model.to("cpu")

    map_model.eval()
    feature_model.eval()

    return map_model, feature_model


def sharedCNN(data, map_model, feature_model):
    #Get map and feature from network
    feature_map = map_model(data)
    feature = feature_model(feature_map.reshape(-1, 512 * 7 * 7))
    #Resize for output
    feature_map = feature_map.reshape(-1, 512, 7 * 7)
    feature = feature.reshape(-1, featurea_length)
    #if not flag_auto:
    #    print(feature_map.size(), feature.size())
    torch.cuda.empty_cache()
    return feature_map, feature



def tar2pdf(inputs, maxx):
    multi_layer = False
    if len(np.shape(inputs)) == 2:
        multi_layer = True
    elif len(np.shape(inputs)) == 1:
        multi_layer = False
    else:
        ValueError("Input data layer error, the target and attribute data must have 2 or 3 deep dimension")

    outputs = []
    if multi_layer:
        for i in range(0, len(inputs)):
            step_output = []
            ttl = 0
            for j in range(0, len(inputs[i])):
                RoundOutput = [0 for n in range(max(nb_attributes))]
                RoundOutput[inputs[i][j]] = 1
                ttl += 1
                step_output.append(torch.Tensor(RoundOutput).to(device))
            outputs.append(step_output)

    else:
        for i in range(0, len(inputs)):
            step_output = [0 for n in range(total_class)]
            
            step_output[inputs[i]] = 1
            
            step_output = step_output
            outputs.append(step_output)
    
    return outputs


def load_data(inputs, batch_size):
    start = 0
    end = 0
    outputs = []
    while 1:
        if end + batch_size >= len(inputs):
            outputs.append(inputs[start:])
            return outputs
        else:
            end += batch_size
            outputs.append(inputs[start: end])
            start = end


def comb_tar_attr(tar, attr):
    outputs = []
    for i in range(0, len(tar)):
        outputs.append([torch.Tensor(tar[i]).to(device), attr[i]])
    return outputs


#Main=================================================================
def main():
    # Input Data from data set ==========================
    train_set, test_set = CUB_load.load_data(dataset_location)
    if not flag_auto:
        print("Data import succeed")
    
    # Data partial and rebuild
    train_image, train_target, train_attr = train_set
    test_image, test_target, test_attr = test_set
    

    # Treatment target and attr to distribution==========
    # Target, attr to pdf for binary cross entropy
    train_target_pdf = tar2pdf(train_target, total_class)
    train_attr_pdf = tar2pdf(train_attr, attr_class)
    
    test_target_pdf = tar2pdf(test_target, total_class)
    test_attr_pdf = tar2pdf(test_attr, attr_class)
    
    # Combine as y data
    train_y = comb_tar_attr(train_target_pdf, train_attr_pdf)
    test_y = comb_tar_attr(test_target_pdf, test_attr_pdf)
    
    # Build to dataset, treat image to map and feature=====
    # Import Images
    train_image_loader = load_data(train_image, batch_size=share_batch_size)
    test_image_loader = load_data(test_image, batch_size=share_batch_size)
    
    # Initial Shared CNN
    map_model, feature_model = build_shared_CNN()

    #Initial Initial Parameters
    train_x = []     #For all train data
    test_x = []      #For all test data
    input_images = torch.Tensor().to(device)
    
    # Main Processing
    with torch.no_grad():
        total = 0
        # Get Feature for Train
        for images in train_image_loader:
            # Read images to cuda
            input_images = torch.Tensor(images).to(device)
            
            # Get Feature
            maps, features = sharedCNN(input_images, map_model, feature_model)
            
            # Build to data set
            for j in range(0, len(maps)):
                train_x.append([maps[j], features[j]])
            # After treat
            total += 1


        total = 0
        # Get Feature for Test
        for images in test_image_loader:
            # Read images to cuda
            input_images = torch.Tensor(images).to(device)

            # Get Feature
            maps, features = sharedCNN(input_images, map_model, feature_model)

            # Build to data set
            for j in range(0, len(maps)):
                test_x.append([maps[j], features[j]])
            # After treat
            total += 1

    if not flag_auto:
        print("Pretrained feature and feature map build succeed")
    

    # Get confident length y data with x size==============
    train_y = train_y[:len(train_x)]
    test_y = test_y[:len(test_x)]

    train_data = map_fea_tar_attr(train_x, train_y)
    test_data = map_fea_tar_attr(test_x, test_y)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = test_batch_size, shuffle = True)

    if not flag_auto:
        print("Data treatment finished")    
        print("Input data structure: Train data size:", len(train_x), "Test data size", len(test_x))
        print("Input: Featur Map Size:", train_x[0][0].size(), "Feature Size:", train_x[0][1].size())
        print("Output: Attribute Size:", len(train_y[0][0]), "Class Size:", len(train_y[0][1]))
        print("Train batch size:", batch_size, "Test batch size:", test_batch_size)

    # DATA INITIAL FINISH==================================

    # Read Model ==========================================
    model = attribute_net().to(device)
    model = nn.DataParallel(model, device_ids=[0,1])
    if flag_auto:
        print(model)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(1, epochs + 1):
        print("Looping epoch = ", epoch)
        start = time.time()
        train(model, train_loader, optimizer, epoch)
        end = time.time()
        t1 = end - start
        
        if epoch % 20 == 0 or flag_all:
            start = time.time()
            test(model, test_loader)
            end = time.time()
            t2 = end - start
            print("Time Usage: Training time", t1, "Testing time", t2)
            
    if (save_model):
        os.system("mkdir Output")
        torch.save(model.state_dict(),"/Output/Model.pt")
    

if __name__ == '__main__':
    main()

