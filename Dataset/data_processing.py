import torch.nn as nn
import torch 

def max_pooling_2X(Input):
    maxPooling = nn.FractionalMaxPool2d(kernel_size=3, 
                                            output_size=(128,128))
        
    a = torch.zeros([1,256,256])
        
    a[:]=Input[:,:,:]
    a = maxPooling(a)
    a = torch.nn.functional.interpolate(
            a.view(1, 1, 128, 128), size=[256,256],  mode='nearest').view(1,256,256)
    return a

def max_pooling_4X(Input):
    maxPooling = nn.FractionalMaxPool2d(kernel_size=3, 
                                            output_size=(128,128))
        
    maxPooling2 = nn.FractionalMaxPool2d(kernel_size=3, 
                                            output_size=(64,64))
    
    a = torch.zeros([1,256,256])
        
    a[:]=Input[:,:,:]
    a = maxPooling(a)
    a = maxPooling2(a)

    a = torch.nn.functional.interpolate(
            a.view(1, 1, 64, 64), size=[128,128],  mode='nearest').view(1,128,128)
    a = torch.nn.functional.interpolate(
            a.view(1, 1, 128, 128), size=[256,256],  mode='nearest').view(1,256,256)
    return a

def max_pooling_8X(Input):
    maxPooling = nn.FractionalMaxPool2d(kernel_size=3, 
                                            output_size=(128,128))
        
    maxPooling2 = nn.FractionalMaxPool2d(kernel_size=3, 
                                            output_size=(64,64))

    maxPooling3 = nn.FractionalMaxPool2d(kernel_size=3, 
                                            output_size=(32,32))
    a = torch.zeros([1,256,256])
        
    a[:]=Input[:,:,:]
    a = maxPooling(a)
    a = maxPooling2(a)
    a = maxPooling3(a)
    a = torch.nn.functional.interpolate(
            a.view(1, 1, 32, 32), size=[64,64],  mode='nearest').view(1,64,64)
    a = torch.nn.functional.interpolate(
            a.view(1, 1, 64, 64), size=[128,128],  mode='nearest').view(1,128,128)
    a = torch.nn.functional.interpolate(
            a.view(1, 1, 128, 128), size=[256,256],  mode='nearest').view(1,256,256)
    return a