from . geometric_extractor import GeometricExtractor
#from perceptual_extratcor import PerceptualNet
from . base import BasicBlock
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch

    
class SuperMeshingNet(torch.nn.Module):

    def __init__(self,num_channel,batch_size):
        super(SuperMeshingNet, self).__init__()
        
        ### ENCODER
        self.geoAttention = GeometricExtractor(num_channel,batch_size)
        self.layer1 = None
        self.delayer1 = None
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
            torch.nn.ConvTranspose2d(in_channels=257,
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
        geo_attention = self.geoAttention(x)
        #texture_attention = self.textureAttention(x)
        ### ENCODER
        encoded0 = self.downscale(x)
        self.layer1 = encoded0
    
        #print("layer 0 shape: ",encoded0.shape)
        encoded1 = self.encoder_layer1(encoded0)
        
        
        encoded2 = self.encoder_layer2(encoded1)
        encoded3 = self.encoder_layer3(encoded2)
        
        ### RESBLOCKS
        
        res=self.res_layer(encoded3)
        
        ### DECODER
        decoded = torch.cat((res,encoded3),1)
        decoded = torch.cat((decoded,geo_attention),1)
        
        decoded = self.decoder_layer3(decoded)
        
        decoded = torch.cat((decoded,encoded2),1)
        
        decoded = self.decoder_layer2(decoded)
        
        decoded = torch.cat((decoded,encoded1),1)
        
        decoded = self.decoder_layer1(decoded)
        
        self.delayer1 = decoded
        
        decoded = torch.cat((decoded,encoded0),1)
        
        decoded = self.updown(decoded)
        
        return decoded, encoded2


