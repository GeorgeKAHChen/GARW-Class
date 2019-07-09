main:
	@python main.py

gmmClu:
	@nohup python Patch_Cluster.py e 0 p 6 b 50000 t 500000 i 2000000 m gmm > Output/6_patch.out &
	@nohup python Patch_Cluster.py e 1 p 7 b 50000 t 500000 i 2000000 m gmm > Output/7_patch.out &
	@nohup python Patch_Cluster.py e 1 p 8 b 50000 t 500000 i 2000000 m gmm > Output/8_patch.out &
	@nohup python Patch_Cluster.py e 0 p 9 b 50000 t 500000 i 2000000 m gmm > Output/9_patch.out &

rwClu:
	@nohup python Patch_Cluster.py e 0 p 6 b 10000 t 500000 i 2000000 m rw > Output/6_patch.out &
	@nohup python Patch_Cluster.py e 1 p 7 b 10000 t 500000 i 2000000 m rw > Output/7_patch.out &
	@nohup python Patch_Cluster.py e 1 p 8 b 10000 t 500000 i 2000000 m rw > Output/8_patch.out &
	@nohup python Patch_Cluster.py e 0 p 9 b 10000 t 500000 i 2000000 m rw > Output/9_patch.out &

testClu:
	@python Patch_Cluster.py p 6 b 500 t 1000 i 2000 m gmm

cnnRW:
	@nohup python easymodel.py -m rw > Output/rw_dcnn.out &

cnnLC:
	@nohup python easymodel.py -m lc > Output/rw_dcnn.out &

gpu:
	@watch -n 0.1 nvidia-smi
	
help:
	@echo "===================================================="
	@echo "GARW: Group Attribute Random Walk Classification"
	@echo ""
	@echo "    A Pytorch Expand Package for Deep Cluster and Non-Linear Classification"
	@echo "    Copyright (c) by KazukiAmakawa, all right reserved."
	@echo "    LICENSE: GPL-3.0"
	@echo "===================================================="
	@echo ""
	@echo "Usage"
	@echo ""
	@echo "    make               MNIST Data set Training"
	@echo "    make gmmClu        Train Patch Classifier with GMM Cluster without mark in different patch"
	@echo "    make rwClu         Train Patch Classifier with GMM Cluster without mark in different patch"
	@echo "    make testClu       Test Patch Classifier Training system"
	@echo "    make cnnRW         Using Random Walk Classfier for classification in CNN training"
	@echo "    make cnnLC         Using Linear Classfier for classification in CNN training"
	@echo "    make gpu           Keep watch gpu using information"
	@echo "    make help          Print this this help message and exit. "
	@echo "    make clean         Clean files in /Output folder"
	@echo ""
	@echo "===================================================="
	@echo ""
	@echo "Source Code: https://www.github.com/KazukiAmakawa/GARW-Class"
	@echo "For more help, please connect this E-mail: GeorgeKahChen@gmail.com"
	@echo ""
	@echo "===================================================="
clean:
	rm -rf Output
	mkdir Output