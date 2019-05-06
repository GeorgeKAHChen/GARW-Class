#=============================================================================
#
#       Group Attribute Random Walk Program
#       parameter.py
#
#       Copyright by KazukiAmakawa, all right reserved
#
#=======================================================
#
#       Intro
#       This file include all parameters in the program. Please change them
#		in this file.
#
#=============================================================================

#Shared CNN, pretrained model
#model_flag = "resnet50"
#model_flag = "resnet18"
#model_flag = "alexnet"
model_flag = "vgg16"
#model_flag = "vgg19"

# Using "cude" for gpu learning(with NVIDIA(R) device), or "cpu" with CPU learning
device = "cuda"

# Training batch size
batch_size = 32		

# Test batch size
test_batch_size = 1

# Total looping size
epochs = 150

# Warning, if you using random walk with parameter, it is necessary to change this loss rate with parameter
lr = 0.2            

# Attribute group
nb_attributes = [10, 16, 16, 16, 5, 16, 7, 16, 12, 16, 16, 15, 4, 16, 16, 16, 16, 6, 6, 15, 5, 5, 5, 16, 16, 16, 16, 5]


momentum = 0.5
seed = 1
log_interval = 10
save_model = False
flag_auto = False

map_size = 14*14
featurea_length = 512