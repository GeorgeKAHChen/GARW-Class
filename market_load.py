#=============================================================================
#
#       Group Attribute Random Walk Program
#       market_load.py
#
#       Copyright by KazukiAmakawa, all right reserved
#
#=======================================================
#
#       Intro
#       This file will load CUB dataset for bird classification
#
#=============================================================================

from libpy import Init
from PIL import Image 
import numpy as np
import torch
import parameter
import time


device = parameter.device

class map_fea_tar_attr(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x_data = x
        self.y_data = y
        self.length = len(x)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.length


def RGBList2Table(InputImage):
    Size = np.shape(InputImage)
    if len(Size) != 3:
        img = []
        img.append(InputImage)
        img.append(InputImage)
        img.append(InputImage)
        return img
    if Size[2] != 3 and Size[0] == 3:
        return InputImage

    RTable = []
    GTable = []
    BTable = []
    for i in range(0, len(InputImage)):
        RLine = []
        GLine = []
        BLine = []
        for j in range(0, len(InputImage[i])):
            RLine.append(InputImage[i][j][0])
            GLine.append(InputImage[i][j][1])
            BLine.append(InputImage[i][j][2])
        RTable.append(RLine)
        GTable.append(GLine)
        BTable.append(BLine)
    return [RTable, GTable, BTable]



def load_data(dir_name, target_size):
    TrainLoc, TrainName = Init.GetSufixFile(dir_name + "/bounding_box_train", ["jpg"]) 
    TestLoc, TestName =  Init.GetSufixFile(dir_name + "/bounding_box_test", ["jpg"]) 

    TrainX = []
    TrainY = []
    TestX = []
    TestY = []
    #Train Treat
    for i in range(0, len(TrainName)):
        img = Image.open(TrainLoc[i])
        img = img.resize(target_size)
        width, height = img.size
        img = list(img.getdata())
        img = [img[i * width:(i + 1) * width] for i in range(height)]
        img = RGBList2Table(img)
        TrainX.append(torch.Tensor(img).to(device))

        name_flag = False
        name_code = 0
        for j in range(0, len(TrainName[i])):
            if TrainName[i][j] == "_":
                name_flag = True
            if name_flag == False:
                name_code = name_code * 10 + int(TrainName[i][j])
            if name_flag == True:
                TrainY.append(name_code)

        if i % 200 == 0 and i != 0:
            print("Import Train Data", i, end = "\r")
    print("Training data import finished")

    #Test Treat
    for i in range(0, len(TestName)):
        if TrainName[i][0] == "-":
            continue
        img = Image.open(TestLoc[i])
        img = img.resize(target_size)
        width, height = img.size
        img = list(img.getdata())
        img = [img[i * width:(i + 1) * width] for i in range(height)]
        img = RGBList2Table(img)
        TestX.append(torch.Tensor(img).to(device))

        name_flag = False
        name_code = 0
        for j in range(0, len(TestName[i])):
            if TestName[i][j] == "_":
                name_flag = True
            if name_flag == False:
                name_code = name_code * 10 + int(TestName[i][j])
            if name_flag == True:
                TestY.append(name_code)

        if i % 200 == 0 and i != 0:
            print("Import Test Data", i, end = "\r")
    print("Test data import finished")


    train_data_class = map_fea_tar_attr(x_data = TrainX, y_data = TrainY)
    test_data_class = map_fea_tar_attr(x_data = TestX, y_data = TestY)
    return train_data_class, test_data_class


if __name__ == '__main__':
    t1 = time.time()
    load_data("Market1501", (64, 128))
    t2 = time.time()
    print(t2-t1)