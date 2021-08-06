#!/usr/bin/env python
# coding: utf-8

import os
from pathlib import Path
import wget
import zipfile
import torchvision
from torchvision import transforms,datasets,models
from torch.utils.data import DataLoader
import torch

# Generate folder function
def directory(directory_path):
    if not os.path.exists(directory_path):
        print(f"Create {directory_path} directory")
        os.mkdir(directory_path)
        print("Done")
    else:
        print(f'Directory {directory_path} already exists, skipping create')

def subfolder(directory_path,target_list):
    directory_contents = os.listdir(directory_path)
    for item in target_list:
        if item not in directory_contents:
            print(f"Create subfolder {item}")
            os.mkdir(os.path.join(directory_path,item))
            print("Done")
        else:
            print(f'Subfolder {item} already exists, skipping create')
            
def folder_generator(folder_path,subfolder_list):
    directory(folder_path)
    subfolder(folder_path,subfolder_list) 


    
# Download resources function    
def download(source, target, filename,zip_file= False):
    if not os.path.exists(target):
        os.mkdir(target)
    targt_file = str(Path(target).joinpath(filename))
    if os.path.exists(targt_file):
        print('file already exists, skipping download')
        return

    print("Downloading from {} to {}".format(source, target))
    wget.download(source, targt_file)
    print("\nDone!")
    if zip_file == True:
        print('Unzipping {}'.format(targt_file))
        zipr = zipfile.ZipFile(targt_file)
        zipr.extractall(target)
        zipr.close()
        print('Done!')   
        
# Testing toolbox
class Toolbox:
    def __init__(self,actual_model,testdir, image_transforms):
        self.actual_model = actual_model
        self.testdir = testdir
        self.image_transforms = image_transforms

    def cal_acc(self,model):
        self.model = model
        self.model.eval()
        test_data = datasets.ImageFolder(root = self.testdir, transform = self.image_transforms)
        testloader = DataLoader(test_data, len(test_data), shuffle=False)

        for images, labels in testloader :
            correct = 0
            y_pred = self.model(images)
            predictions = torch.max(y_pred, 1)[1]
            correct += (predictions == labels).sum()
        accuracy = correct / len(test_data) 
        return accuracy
    
    # File test
    def file_test(self,directory_path):
        if os.path.isfile(directory_path):
            print("Pass File Test")
        else:
            print("No File Found,Please Try Again")
        
    # Accuracy Test        
    def accuracy_test(self,student_model):
        print("Testing Accuracy ...")
        if student_model == None:
            print("No Accuracy Test for This Question")
        elif self.cal_acc(self.actual_model) == self.cal_acc(student_model):
            print("Pass Accuracy Test")
        else:
            print("Fail,Please Try Again") 

    # Combine test        
    def test(self,model_path,model_function=None):
        self.file_test(model_path)
        self.accuracy_test(model_function)  
