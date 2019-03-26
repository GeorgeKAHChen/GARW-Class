############################################################
#
#		GARW-Class
#		GARW.py
#		Copyright(c) KazukiAmakawa, all right reserved.
#
############################################################
from keras import backend as K
from keras.layers import Layer

def RGBList2Table(InputImage):
	import numpy as np
	Size = np.shape(InputImage)
	if len(Size) != 3:
		print("InputError: RGBList2Table function need input image with [[[R,G,B] * width] * height] parameter. Your input may RGB tabled image or grey image")
		return [[[-1]], [[]], [[]]]
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
	return np.array([RTable, GTable, BTable])


def GetDistance(Inps, Kernel, Distance, Parameter):
	import math
	Inp1 = Inps[2]
	Inp2 = Inps[3]
	Total = 0
	for i in range(0, len(Inp1)):
		Total += Parameter[0] * pow(Inp1[i] - Inp2[i], 2)
	return [Inps[0], Inps[1], math.exp(-Total)]


def RandomLayer(InputData, Group, Para = ["Gaussian", "Euclid", False, [0.01]]):
	import multiprocessing
	from functools import partial
	import numpy as np

	#Initial definition parameter
	Kernel = Para[0]
	Distance = Para[1]
	FlagTest = Para[2]
	ParaVals = Para[3]

	#Determine group classification or mode classification
	NoGroup = False
	if len(Group) == 0:
		NoGroup = True
	else:
		if len(Group) != len(InputData[0]):
			print("InputError: The length of labeled data must have same length with the length of Group")
			return
	
	#Partial data into XL, XU
	XL, XU = InputData
	SizeL = len(XL)
	SizeU = len(XU)

	#Initial Probability transformation matirx
	Pul = [[0 for n in range(SizeL)] for n in range(SizeU)]
	Puu = [[0 for n in range(SizeU)] for n in range(SizeU)]
	Psum = [0 for n in range(SizeU)]
	
	#Build multiprocessing calculation matrix
	Points = []
	for i in range(0, SizeU):
		for j in range(0, SizeL):
			Points.append([i, j, XU[i], XL[j]])

	for i in range(0, SizeU):
		for j in range(i + 1, SizeU):
			Points.append([i, j, XU[i], XU[j]])

	#Build multiprocessing function with partial function
	#Calculate all distences
	PartDis = partial(GetDistance, Kernel= Kernel, Distance = Distance, Parameter = ParaVals)
	pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
	results = pool.map(PartDis, Points)
	pool.close()
	pool.join()

	#Get Pul
	for i in range(0, SizeU * SizeL):
		RealX = results[i][0]
		RealY = results[i][1]
		Pul[RealX][RealY] = results[i][2]
		Psum[RealX] += results[i][2]


	#Get symmetric Puu
	for i in range(SizeU * SizeL, len(results)):
		RealX = results[i][0]
		RealY = results[i][1]
		Puu[RealX][RealY] = results[i][2]
		Puu[RealY][RealX] = results[i][2]
		Psum[results[i][0]] += results[i][2]
		Psum[results[i][1]] += results[i][2]

	for i in range(0, len(Psum)):
		if Psum[i] == 0:
			Psum[i] += 1

	#Calculate, negative for Puu = (I-Puu)
	for i in range(0, SizeU):
		for j in range(0, SizeL):
			Pul[i][j] /= Psum[i]
		for j in range(0, SizeU):
			Puu[i][j] /= -Psum[i]
		Puu[i][i] = 1

	#Solving programming
	Xul = np.array(np.linalg.inv(np.matrix(Puu)) * np.matrix(Pul))

	#For Output and Decision
	Decision = [0 for n in range(SizeU)]
	for i in range(0, SizeU):
		if NoGroup:
			#Classification for no group, find the maximum sign
			MaxVal = 0
			for j in range(0, SizeL):
				if Xul[i][j] >= MaxVal:
					MaxVal = Xul[i][j]
					Decision[i] = j
		
		else:
			#Classification for group data, get group probability
			SingleGroup = [0 for n in range(max(Group) + 1)]
			for j in range(0, SizeL):
				SingleGroup[Group[j]] += Xul[i][j]

			#Find maximum group
			print(Xul[i])
			print(SingleGroup)
			MaxVal = 0
			for j in range(0, len(SingleGroup)):
				if SingleGroup[j] > MaxVal:
					MaxVal = SingleGroup[j]
					Decision[i] = j

	if FlagTest:
		from libpy import Init
		Init.ArrOutput(Puu)
		Init.ArrOutput([[]])
		Init.ArrOutput(Pul)
		Init.ArrOutput([[]])
		Init.ArrOutput(Xul)
		Init.ArrOutput([[]])
		Init.ArrOutput([Decision])

	return Decision




if __name__ == '__main__':
	#For test every layer and functions
	#This is not the begining of the program, it is just for the test!!!
	#DO NOT RUN DIRECTLY IT IF YOU ARE NOT CONTRIBUTER
	print(RandomLayer([[[1, 3], [4, 5], [3, 5]], [[2, 4], [2, 6], [2, 8] ,[4, 6]]], [0, 1, 1],  ["Gaussian", "Euclid", True, [0.01]]))
"""





