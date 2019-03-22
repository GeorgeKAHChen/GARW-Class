from copy import deepcopy
import GARW
from libpy import Init
import numpy as np
from PIL import Image
from scipy import misc


RWOutput = True
SignImg = 1


def GetFiles(FileDir, height, width): 
	import os

	RootFolder = ""
	Catagory = []
	XLs = []
	XUs = []
	XLSign = []
	XUSign = []
	ttl = -1
	
	for root, dirs, files in os.walk(FileDir):
		#Get direction and location
		if ttl == -1:
			RootFolder = root
			Catagory = deepcopy(dirs)
			ttl += 1
			continue

		for i in range(0, len(files)):
			FileName = RootFolder + "/" + Catagory[ttl] + "/" + files[i]
			img = GARW.RGBList2Table(misc.imresize(np.array(Image.open(FileName)), (height, width, 3)))
			#feature = model.predict([img])[len(feature) - 1]
			feature = img[0][0]			#Test Program without A3MCNN feature
			#print(feature)
			#Part XL, XU		
			if i >= 0 and i < SignImg:
				XLs.append(feature)
				XLSign.append(ttl)
			else:
				XUs.append(feature)
				XUSign.append(ttl)

		ttl += 1

	if len(XLs) == 0 or len(XUs) == 0:
		print("Input Parameter Error: SignImg")
		return
	#RW Classification	
	XURWSign = GARW.RandomLayer([XLs, XUs], kernel = "Gaussian", distance = "Euclid", para = [], method = "")
	
	Succeed = 0
	Output = []
	for i in range(0, len(XURWSign)):
		if XLSign[XURWSign[i]] == XUSign[i]:
			Succeed += 1
	
		if RWOutput == True:
			Output.append([XURWSign[i], XLSign[XURWSign[i]], XUSign[i], 1 if XLSign[XURWSign[i]] == XUSign[i] else 0])

	if RWOutput == True:
		Init.ArrOutput([Catagory])
		Init.ArrOutput([XLSign])
		Init.ArrOutput(Output)
		Init.ArrOutput([[Succeed, len(XURWSign), Succeed/len(XURWSign)]])

	print("\n\nRandom Classification finished, Total Image:" + str(len(XURWSign)) + ", Succeed: " + str(Succeed) + ".\n Accuracy: " + str(Succeed/len(XURWSign)))
	

GetFiles(FileDir = "./Input", height = 448, width = 448)







