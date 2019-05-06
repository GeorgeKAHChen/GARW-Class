#=============================================================================
#
#       Group Attribute Random Walk Program
#       Label.py
#
#       Copyright by KazukiAmakawa, all right reserved
#
#=======================================================
#
#       Intro
#       This is a small project we build for mark in pedestrin re-identification
#       mark. You can read the detail usage in the README.md file
#
#=============================================================================
import os
from libpy import Init
import numpy as np
from PIL import Image
import pygame
from pygame.locals import *
import time

build_checklist = False

def ReadMarkList(MarkLoc):
    return

def BuildCheckList(Files, Loc):
    return

def ReadCheckList(CheckLoc):
    return

def SaveCheckList(List):
    return

def main(FileDir, MarkLoc, CheckLoc):
    Loc, Files = Init.GetSufixFile(FileDir, ["png", "jpg"])
    print(Files)
    print(Loc)

    if not os.system.exists(MarkLoc):
        ValueError("MarkList are not exist, please add MarkList before marking")

    Marks = ReadMarkList(MarkLoc)

    if not os.system.exists(CheckLoc):
        build_checklist = True

    if build_checklist:
        BuildCheckList(Files, Loc, CheckLoc)

    List = ReadCheckList(CheckLoc)

    for i in range(0, len(List)):
        if len(List[i]) == 1:
            List[i].append(Mark(List[i]))

        if save_flag == True:
            save_flag = False
            SaveCheckList(List)

if __name__ == '__main__':
    main("/Users/kazukiamakawa/Desktop/tem", "./Output/MarkList", "./Output/CheckList")