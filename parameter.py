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
#       in this file.
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

# share CNN calculation batch size
share_batch_size = 200

# Total looping size
epochs = 5000

# Warning, if you using random walk with parameter, it is necessary to change this loss rate with parameter
lr = 0.001

# Attribute group
nb_attributes = [10, 16, 16, 16, 5, 16, 7, 16, 12, 16, 16, 15, 4, 16, 16, 16, 16, 6, 6, 15, 5, 5, 5, 16, 16, 16, 16, 5]

# Attribute group
total_class = 200

# Dataset location
dataset_location = "CUB_test"

# SGD momentum
momentum = 0.5

# Random value seed
seed = 1

# How many batches to wait before logging training status
log_interval = 10

# Model saving flag
save_model = False

# Running the processing without print information
flag_auto = False

# Size of map in feature map
map_size = 7 * 7

# Size of final feature
featurea_length = 1000 







