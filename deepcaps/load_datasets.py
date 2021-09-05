import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os


def load_tiny_imagenet(data_path,normal):
    numall=210000
    example_ni1 = os.path.join(data_path, 'tr_im.nii.gz')
    n1_img = nib.load(example_ni1)
    n1_header = n1_img.header
    x_train = n1_img.get_fdata()#512*512*100
    
    example_ni1 = os.path.join(data_path, 'tr_mask.nii.gz')
    n1_img = nib.load(example_ni1)
    n1_header = n1_img.header
    train_y  = n1_img.get_fdata()#512*512*100
    
    train_d=normalization(x_train)
    
    if (normal==1):
         train_data_patch1,train_data_patch2,train_data_y=patch_maker_normal(train_d,train_y)
    else:
         train_data_patch1,train_data_patch2,train_data_y=patch_maker(train_d,train_y)
            
   
    cat_y= np.zeros((numall,4),dtype=np.float32)
    for j in range(numall):
          cat_y[j,int(train_data_y[j])] = 1 
    return train_data_patch1,train_data_patch2,cat_y
    
    
    
    
def normalization (train_data):
    normal_d=np.zeros([512,512,100], dtype = 'float32') 
    for i in range(100):
        d=train_data[:,:,i]
        norm = (d - np.min(d)) / (np.max(d) - np.min(d))
        normal_d[:,:,i]=norm
    return normal_d


def patch_maker(x_train,y_train):
    numall=210000
    X_train=np.zeros((numall,33,33),dtype=np.float32)
    X_train2=np.zeros((numall,15,15),dtype=np.float32)
    Y_train=np.zeros((numall,1),dtype=np.float32)
    input_dim=33
    num_patch=0
    
    for num_patient in range(0,100):
        for i in range(int((input_dim)/2),y_train.shape[0]-int((input_dim)/2)-1):
                for j in range(int((input_dim)/2),y_train.shape[1]-int((input_dim)/2)-1):
                     n=np.random.rand(1)
                     m=np.random.rand(1)
                     if(n<0.15 and m<0.1):
                       X_train[num_patch,:,:]=x_train[i-16:i+17,j-16:j+17,num_patient]
                       X_train2[num_patch,:,:]=x_train[i-int((15)/2):i+int((15)/2)+1,j-int((15)/2):j+int((15)/2)+1,num_patient]
                       Y_train[num_patch]=y_train[i,j,num_patient]
                       num_patch=num_patch+1
                       if (num_patch==numall):
                           return X_train,X_train2,Y_train
                       
def patch_maker_normal(x_train,y_train):
    print("****load data normally****")
    numall=210000
    q=[3*numall/8,2*numall/8,2*numall/8,numall/8]
    X_train=np.zeros((numall,33,33),dtype=np.float32)
    X_train2=np.zeros((numall,15,15),dtype=np.float32)
    Y_train=np.zeros((numall,1),dtype=np.float32)
    input_dim=33
    num_patch=0
    
    num_class=[0,0,0,0]
    num_patient=0
    
    while(num_patch<numall):
        for i in range(int((input_dim)/2),y_train.shape[0]-int((input_dim)/2)-1):
                for j in range(int((input_dim)/2),y_train.shape[1]-int((input_dim)/2)-1):
                    mm=int(y_train[i,j,num_patient])
                    if (num_class[mm]<q[mm]):
                         n=np.random.rand(1)
                         if(n<0.15):
                           X_train[num_patch,:,:]=x_train[i-16:i+17,j-16:j+17,num_patient]
                           X_train2[num_patch,:,:]=x_train[i-int((15)/2):i+int((15)/2)+1,j-int((15)/2):j+int((15)/2)+1,num_patient]
                           Y_train[num_patch]=y_train[i,j,num_patient]
                           num_class[mm]=num_class[mm]+1
                           num_patch=num_patch+1
                           print(num_patient,mm,num_class[mm],num_patch)
                           if(num_patient==99):
                               num_patient=0
        num_patient=num_patient+1
                           
                           
    return X_train,X_train2,Y_train