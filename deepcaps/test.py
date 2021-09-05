# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 09:33:41 2020

@author: Admin
"""
from keras import optimizers
import keras.callbacks as callbacks
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model
from utils import margin_loss, margin_loss_hard, CustomModelCheckpoint

from twowaydeep import DeepCapsNetTwoPath
from keras import backend as K
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
from sklearn import metrics


def test(eval_model, data):

    (x_test,x_test2)  = data
    eval_model.load_weights(args.save_dir+"/best_weights_2.h5")
    a1, b1 = eval_model.predict([x_test,x_test2])
    return a1, b1

def normalization (train_data):
    d=train_data[:,:]
    norm = (d - np.min(d)) / (np.max(d) - np.min(d))
    return norm

def testDataGenerator(numOfP):
    #make data for 1 patient
    data_path=r'E:\\Projects\\reyhaneh\\covid19\\dataset\\'
    example_ni1 = os.path.join(data_path, 'tr_im.nii.gz')
    n1_img = nib.load(example_ni1)
    n1_header = n1_img.header
    x_test = n1_img.get_fdata()#512*512*100
    x_= x_test[:,:,numOfP]
    x=normalization(x_)
    
    
    example_ni1 = os.path.join(data_path, 'tr_mask.nii.gz')
    n1_img = nib.load(example_ni1)
    n1_header = n1_img.header
    y_test  = n1_img.get_fdata()#512*512*100
    y= y_test[:,:,numOfP]
    
    numall=229441
    X_test=np.zeros((numall,33,33),dtype=np.float32)
    X_test2=np.zeros((numall,15,15),dtype=np.float32)
    Y_test=np.zeros((numall,1),dtype=np.float32)
    
    num_patch=0
    input_dim=33
    for i in range(int((input_dim)/2),y.shape[0]-int((input_dim)/2)-1):#32-123
      for j in range(int((input_dim)/2),y.shape[1]-int((input_dim)/2)-1):#32-208
          X_test[num_patch,:,:]=x[i-16:i+17,j-16:j+17]
          X_test2[num_patch,:,:]=x[i-int((15)/2):i+int((15)/2)+1,j-int((15)/2):j+int((15)/2)+1]
          Y_test[num_patch]=y[i,j]
          num_patch=num_patch+1
         
    return X_test,X_test2,Y_test


def discesore(seg,gt):
 	num=0
 	for i in range(479):
         for j in range(479):
             if(seg[i][j]==gt[i][j]):
                 num=num+1        
 	return num/(479*479)

def discesore_c1(seg,gt):
    num=0
    all_p = 0
    for i in range(479):
        for j in range(479):
            if(seg[i][j]==gt[i][j])and(gt[i][j]==1):
                num=num+1
            if (gt[i][j]==1):
                all_p = all_p + 1
    if num == all_p:
        d = 1
    elif all_p == 0:
        d = -1
    else:
        d = num/(all_p)
        print("no")
    print(num)
    print(all_p)                  
    return d
def discesore_c2(seg,gt):
    num=0
    all_p = 0
    for i in range(479):
        for j in range(479):
            if(seg[i][j]==gt[i][j])and(gt[i][j]==2):
                num=num+1
            if (gt[i][j]==2):
                all_p = all_p + 1
    if num == all_p:
        d = 1
    elif all_p == 0:
        d = -1
    else:
        d = num/(all_p)
        print("no")
    print(num)
    print(all_p)                    
    return d
def discesore_c3(seg,gt):
    num=0
    all_p = 0
    for i in range(479):
        for j in range(479):
            if(seg[i][j]==gt[i][j])and(gt[i][j]==3):
                num=num+1
            if (gt[i][j]==3):
                all_p = all_p + 1
    if num == all_p:
        d = 1
    elif all_p == 0:
        d = -1
    else:
        d = num/(all_p) 
        print("no")
    print(num)
    print(all_p)                     
    return d

def discesore2(seg,gt):
    num=0
    all_p = 0
    for i in range(479):
        for j in range(479):
            if(seg[i][j]==gt[i][j])and(gt[i][j]!=0):
                num=num+1
            if (gt[i][j]!=0):
                all_p = all_p + 1
    return num/(all_p)

def mytest(numP,eval_model):
      x_test,x_test2,y_test=testDataGenerator(numP)
     
      x_test=x_test.reshape(-1, 33, 33, 1)
      x_test2=x_test2.reshape(-1, 15, 15, 1)
         
     
      yTest = np.zeros((y_test.shape[0],4))
      for j in range(yTest.shape[0]):
            yTest[j,int(y_test[j])] = 1 
        
     
      #plot gt
      wid=479
      h=479  
      gt=y_test.reshape(wid,h)
      plt.imshow(gt)
      plt.show()
     
      #calculate predict
      a1,b1=test(eval_model, (x_test,x_test2))
      pred = np.around(a1)
      pred1 = np.argmax(pred.reshape(y_test.shape[0],4)[:,1:3],axis = 1)
      # bb1 = np.argmax(a1,1)
      # bb = np.reshape(bb1,(wid,h))
      #plot predict
      pred=pred1.reshape(wid,h)
      plt.imshow(pred)
      plt.show()
     
       #f1score
      f1 = metrics.f1_score(y_test,pred1,average='micro')
      print(f1)
       
      a1 = metrics.accuracy_score(y_test,pred1)
      print(a1)
     
      # #dicescore
      # d=discesore(pred,gt)
      # print(d)
     
      # #dice2
      # d1=discesore2(pred, gt)
      # print(d1)
      # print("*************class1*************")
      # #dice class 1
      # d3 = discesore_c1(pred,gt)
      # print(d3)
      # print("*************class2*************")
      # #dice class 2
      # d4 = discesore_c2(pred,gt)
      # print(d4)
      # print("*************class3*************")
      # #dice class 3
      # d5 = discesore_c3(pred,gt)
      # print(d5)
     
    
     
     
     

class args:
    numGPU = 1
    epochs = 30
    batch_size = 128
    lr = 0.001
    lr_decay = 0.96
    lam_recon = 0.4
    r = 3
    routings = 3
    shift_fraction = 0.1
    debug = False
    digit = 5
    save_dir = 'model/covid19'
    t = False
    w = None
    ep_num = 0
    dataset = "my_data"



model, eval_model = DeepCapsNetTwoPath((33,33,1),(15,15,1), n_class=4, routings=args.routings)
for i in range(90,100):
    print("patient",i)
    mytest(i,eval_model)


    
    
##############################################