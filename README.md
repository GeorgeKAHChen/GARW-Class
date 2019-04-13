# Group Attributeion Random Walk Classification Method
This is the main program of Group Attribution Random Walk Classification Method in image classification and pedestrian re-identificaiton. This method mainly combined Random Walk, Neural Network and Attribution Classificaion together.

## Usage
WARNING: This project may cannot be build on Windows. It is Unix only.


## File Structure 
### Folder ./ 
GARW.py: Group Attribute Random Walk Network Structure Function

main.py: main function to build a training/testing processing

makefile: Parameter setting files and actual runing file

P.S. You can using the main.py file directly, however, it is necessary for you to detemine every necessary parameter. Here are table of parameter introduction

-t runing the program with training model
-r runing the program with test model(after model training)

### Folder ./Output
This folder will include every output files 

### Folder ./Input
This folder will include input files if you don't set your input files and try to run the project directly

### Folder ./libpy
This folder include some image processing fuction I always used

### Folder ./model
This folder will include the output network structure after training.

## Citation


## Connect us
Of course, you can use issue to tell us your problems our feedback bugs. Also, you can connect us with

Telegram: https://t.me/KazukiAmakawa

Mail: KazukiAmakawa@gmail.com


## Version Information
0.0.0: initial version added readme and sample files<br/>
0.0.1: Program build, RW function build<br/>
0.0.2: Added GARW.RGBList2Table and A3MTest.py for fine-gained feature testing<br/>
0.1.0: Added Group classification in RW fucntion<br/>
0.1.1: Fixed unexcepted input error in GARW.RGBList2Table<br/>
0.1.2: Fixed a bug in RW function, added version information in main readme file<br/>
0.2.0: Added A3Model (Source Code: https://github.com/iamhankai/attribute-aware-attention) 
in keras2 py3, not fixed bug and test<br/>
0.2.1: A3MDebug, Fixed Bug, added License<br/>
0.2.2: Add Ideneity Layer Test Files
0.3.0: Added initial random layer without training part
0.3.1: Added keras source code for own layer testing
0.3.2: Distance is finish with ref notes, also deleted keras ref
0.3.3: Non-linear Layer finished, need test
0.3.3: (A Stable Versiom in Non-Linear Classification) fixed bug in tf.exp(tf.math.exp -> tf.exp)
0.4.0: Added Random Walk Layer for testing, fixed bug in layer shape, changed makefile for fast using, changed filename
0.4.1: Fixed filename and several bug, added version information into README.md

## LICENSE
GNU GENERAL PUBLIC LICENSE Version 3