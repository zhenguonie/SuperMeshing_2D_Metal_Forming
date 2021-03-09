from .base import BasicBlock
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch


class GeometricExtractor(torch.nn.Module):

    def __init__(self,num_channel,batch_size):
        super(GeometricExtractor, self).__init__()
        
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
        
        self.encoder_layer2=nn.Sequential( 
            torch.nn.Conv2d(in_channels=num_channel[1],
                            out_channels=num_channel[2],
                            kernel_size=(6, 6),
                            stride=(2, 2),
                            padding=2),
            
            nn.BatchNorm2d(num_channel[2]),
            
            nn.ReLU()
        )
        
        self.encoder_layer3=nn.Sequential(
            torch.nn.Conv2d(in_channels=num_channel[2],
                            out_channels=num_channel[3],
                            kernel_size=(4, 4),
                            stride=(2, 2),
                            padding=1),
            
            nn.BatchNorm2d(num_channel[3]),
            
            nn.ReLU()
        )
        
        self.encoder_layer4=nn.Sequential(
            torch.nn.Conv2d(in_channels=num_channel[3],
                            out_channels=1,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=1),
            
            nn.ReLU()
        )
        
        ### RESBLOCK
        self.res_layer = self._make_layer(BasicBlock, num_channel[3], num_channel[3], 6)
        
        ### DECODER
        
        self.decoder_layer3=nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=num_channel[4],
                            out_channels=num_channel[2],
                            kernel_size=(4,4),
                            stride=(2,2),
                            padding=1),
            
            nn.BatchNorm2d(num_channel[2]),
            
            nn.ReLU()
        )
        
        self.decoder_layer2=nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=num_channel[3],
                            out_channels=num_channel[1],
                            kernel_size=(6,6),
                            stride=(2,2),
                            padding=2),
            
            nn.BatchNorm2d(num_channel[1]),
            
            nn.ReLU()
        )
        
        self.decoder_layer1=nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=num_channel[2],
                            out_channels=num_channel[0],
                            kernel_size=(8,8),
                            stride=(2,2),
                            padding=3),
            
            nn.BatchNorm2d(num_channel[0]),
            
            nn.ReLU()
        )
        
        self.updown=nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=num_channel[1],
                                     out_channels=int(num_channel[1]/4),
                                     kernel_size=(4, 4),
                                     stride=(2, 2),
                                     padding=0),
            
            nn.BatchNorm2d(int(num_channel[1]/4)),
            
            nn.ReLU(),
            
            torch.nn.Conv2d(in_channels=int(num_channel[1]/4),
                                     out_channels=1,
                                     kernel_size=(5, 5),
                                     stride=(1, 1),
                                     padding=1)
        )
            
            

    def _make_layer(self, block, inplanes, planes, blocks, stride=1): 

        layers = []

        for i in range(0, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        
        
        #Main feature
        ### ENCODER
        encoded0 = self.downscale(x)
        #print("layer 0 ",encoded0.shape)
        encoded1 = self.encoder_layer1(encoded0)
        #print("layer 1 ",encoded1.shape)
        encoded2 = self.encoder_layer2(encoded1)
        #print("layer 2 ",encoded2.shape)
        encoded3 = self.encoder_layer3(encoded2)
        #print("layer 3 ",encoded3.shape)
        
        ### RESBLOCKS
        
        res=self.res_layer(encoded3)
        #print("layer res ",res.shape)
        encoded4 = self.encoder_layer4(res)
        #print("layer 4 ",encoded4.shape)
        return encoded4