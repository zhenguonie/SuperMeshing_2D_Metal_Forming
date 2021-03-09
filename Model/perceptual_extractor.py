from base import BasicBlock
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch

class PerceptualNet(torch.nn.Module):

    def __init__(self,num_channel,batch_size):
        super(PerceptualNet, self).__init__()
        
        ### ENCODER
        
        self.downscale=nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                           out_channels=num_channel[0],
                           kernel_size=(10,10),
                           stride=(2,2),
                           padding=4),
            
            nn.BatchNorm2d(num_channel[0]),
            
            nn.ReLU()
        )
        
        self.encoder_layer1=nn.Sequential(
            torch.nn.Conv2d(in_channels=num_channel[0],
                            out_channels=num_channel[1],
                            kernel_size=(8, 8),
                            stride=(2, 2),
                            padding=3),
            
            nn.ReLU(),
            
            nn.BatchNorm2d(num_channel[1])
        )
        
        
            

    def _make_layer(self, block, inplanes, planes, blocks, stride=1): 

        layers = []

        for i in range(0, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        
        ### ENCODER
        encoded0 = self.downscale(x)
        
        encoded1 = self.encoder_layer1(encoded0)
        
        
        return encoded1