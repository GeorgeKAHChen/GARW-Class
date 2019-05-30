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

model_flag = "vgg16"            # Shared CNN, pretrained model
device = "cuda"                 # Using "cude" for gpu learning(with NVIDIA(R) device), or "cpu" with CPU learning
batch_size = 32                 # Training batch size
test_batch_size = 1             # Test batch size
share_batch_size = 200          # share CNN calculation batch size
epochs = 20000                  # Total looping size
lr = 0.01                       # Warning, if you using random walk with parameter, it is necessary to change this loss rate with parameter
nb_attributes = [10, 16, 16, 16, 5, 16, 7, 16, 12, 16, 16, 15, 4, 16, 16, 16, 16, 6, 6, 15, 5, 5, 5, 16, 16, 16, 16, 5]
                                # Attribute group
total_class = 200               # Class group size
dataset_location = "CUB_test"
                                # Dataset location
momentum = 0.5                  # SGD momentum
seed = 1                        # Random value seed
log_interval = 10               # How many batches to wait before logging training status
save_model = True               # Model saving flag
flag_auto = True                # Running the processing without print information
map_size = 7 * 7                # Size of map in feature map
featurea_length = 512           # Size of final feature
flag_all = True                 # Test in every round?






