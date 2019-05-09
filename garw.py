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
flag_auto = parameter.flag_auto
map_size = parameter.map_size
featurea_length = parameter.featurea_length
dataset_location = parameter.dataset_location
nb_attributes = parameter.nb_attributes
total_class = parameter.total_class

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
        #self.map2attr = NLRWClass.NLRWDense(input_features = map_size, output_features = attr_class, work_style = "RW", UL_distant = 0.05, UU_distant = 0.05, device = device)
        #self.attr2final = NLRWClass.NLRWDense(input_features = attr_class, output_features = total_class, work_style = "RW", UL_distant = 0.1, UU_distant = 0.1, device = device)
        #self.map2final = NLRWClass.NLRWDense(input_features = featurea_length, output_features = total_class, work_style = "RW", UL_distant = 0.1, UU_distant = 0.1, device = device)
        self.map2attr = nn.Linear(map_size, attr_class, bias = False)
        self.attr2final = nn.Linear(attr_class, total_class, bias = False)
        self.map2final = nn.Linear(featurea_length, total_class, bias = False)

    def forward(self, x):
        #Input [[None, [512, 196]], [None, 512]]
        attr_map, features = x
        attr_dis = torch.Tensor().to(device)
        for i in range(0, len(attr_map)):
            attr_pdf = self.map2attr(attr_map[i])
            attr_pdf = F.softmax(attr_pdf)
            attr_sum = torch.sum(attr_pdf.t(), dim = 1)
            attr_dis = torch.cat([attr_dis, attr_sum])

        attr_dis = attr_dis.reshape(-1, attr_class)
        attr_dis = attr_dis / attr_class

        attr_final = self.attr2final(attr_dis)
        attr_final = F.softmax(attr_final)

        map_final = self.map2final(features)
        map_final = F.softmax(map_final)
        
        final = (attr_final + map_final) / 2
        
        return final, attr_dis





#Main Training and Testing ===========================================
def train(model, trains, optimizer, epoch):
    model.train()
    optimizer.zero_grad()
    data, (train_target_pdf, train_attr_pdf) = trains
    final, attr_dis = model(data)
    loss1 = F.binary_cross_entropy(final, train_target_pdf)
    loss2 = F.binary_cross_entropy(attr_dis, train_attr_pdf)
    loss = loss1 + loss2
    loss.backward()
    optimizer.step()
    print(loss1, loss2, loss)
    """
    if not flag_auto:
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t\t\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.item()), end = "\r")
    if not flag_auto:
        print()
    """


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
    input("6")
    #Get map and feature from network
    feature_map = map_model(data)
    feature = feature_model(feature_map.reshape(-1, 512 * 7 * 7))
    input("7")  
    #Resize for output
    feature_map = feature_map.reshape(-1, 512, 7 * 7)
    feature = feature.reshape(-1, featurea_length)
    if not flag_auto:
        print(feature_map.size(), feature.size())
    input("10")
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
            step_output = np.array([0 for n in range(attr_class)])
            
            tmp = 0
            for j in range(0, len(inputs[i])):
                step_output[tmp + inputs[i][j]] = 1
                tmp += nb_attributes[j]
            step_output = step_output / attr_total
            
            step_output = step_output
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


#Main=================================================================
def main():
    #Input Data from data set ==========================
    train_set, test_set = CUB_load.load_data(dataset_location)
    if not flag_auto:
        print("Data import succeed")
    
    #Data partial and rebuild
    train_image, train_target, train_attr = train_set
    test_image, test_target, test_attr = test_set
    


    #Treatment target and attr to distribution==========
    """
    train_target = train_target[0:225]
    train_attr = train_attr[0:225]
    test_target = test_target[0:175]
    test_attr = test_attr[0:175]
    """
    #Target to pdf for binary cross entropy
    train_target_pdf = tar2pdf(train_target, total_class)
    train_attr_pdf = tar2pdf(train_attr, attr_class)
    
    test_target_pdf = tar2pdf(test_target, total_class)
    test_attr_pdf = tar2pdf(test_attr, attr_class)
    
    train_image_loader = load_data(train_image, batch_size=share_batch_size)
    test_image_loader = load_data(test_image, batch_size=share_batch_size)
    



    #Build to dataset, treat image to map and feature=====
    #Initial Shared CNN
    map_model, feature_model = build_shared_CNN()

    #Initial Initial Parameters
    train_data = []     #For all train data
    test_data = []      #For all test data
    input_images = torch.Tensor().to(device)
    

    with torch.no_grad():
        total = 0
        #Get Feature for Train
        for images in train_image_loader:
            #Read images to cuda
            input_images = torch.Tensor(images).to(device)
            
            #Get Feature
            maps, features = sharedCNN(input_images, map_model, feature_model)

            #Build to data set
            for j in range(0, len(maps)):
                train_data.append([
                                    [maps[j], 
                                     features[j]
                                    ], 
                                    [train_target_pdf[total * share_batch_size + j],
                                     train_attr_pdf[total * share_batch_size + j]
                                    ]
                                  ])
            #After treat
            total += 1


        total = 0
        #Get Feature for Test
        for images in test_image_loader:
            #Read images to cuda
            input_images = torch.Tensor(images).to(device)

            #Get Feature
            maps, features = sharedCNN(input_images, map_model, feature_model)

            #Build to data set
            for j in range(0, len(maps)):
                test_data.append([
                                    [maps[j], 
                                     features[j]
                                    ], 
                                    [test_target_pdf[total * share_batch_size + j],
                                     test_attr_pdf[total * share_batch_size + j]
                                    ]
                                  ])
            #After treat
            total += 1


    if not flag_auto:
        print("Pretrained feature and feature map build succeed")

    if not flag_auto:
        print("Training data structure: Total: ", len(train_data), "Map Length:", len(train_data[0][0][0]), "Map Size:", len(train_data[0][0][0][0]), "Feature Size:", len(train_data[0][0][1]), "Class Size:", len(train_data[0][1][0]), "Attribute Size:", len(train_data[0][1][1]))
        print("Testing data structure: Total: ", len(test_data), "Map Length:", len(test_data[0][0][0]), "Map Size:", len(test_data[0][0][0][0]), "Feature Size:", len(test_data[0][0][1]), "Class Size:", len(test_data[0][1][0]), "Attribute Size:", len(test_data[0][1][1]))


    
    
    model = attribute_net().to(device)
    model = nn.DataParallel(model,device_ids=[0,1])
    if flag_auto:
        print(model)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)


    for epoch in range(1, epochs + 1):
        print("Looping epoch = ", epoch)
        start = time.time()
        train(model, trains, optimizer, epoch)
        end = time.time()
        t1 = end - start

        #test(model, device, train_loader)
        start = time.time()
        #test(model, device, test_loader)
        end = time.time()
        t2 = end - start
        print("Time Usage: Training time", t1, "Testing time", t2)
        time.sleep(2)
        
    if (save_model):
        os.system("mkdir Output")
        torch.save(model.state_dict(),"/Output/Model.pt")
    

if __name__ == '__main__':
    main()

