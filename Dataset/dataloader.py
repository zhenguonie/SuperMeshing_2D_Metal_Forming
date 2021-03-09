from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.utils.data as data
import random
from . import data_processing
import numpy as np
import torch



class FormDataset(data.Dataset):
    def __init__(self,Inputs,Targets):
        self.Inputs=Inputs
        self.Targets=Targets
        
        
    def __getitem__(self,index):
        Input, Target=self.Inputs[index,:,:,:], self.Targets[index,:,:]
        
        Input=torch.from_numpy(Input).float()
        Target=torch.from_numpy(Target).float()
        
        #Change the scaling methods to change to scale factor
        a = data_processing.max_pooling_2X(Input)
        
        b = torch.zeros([1,256,256])
        b[0] = Target
        
        return a, b
    
    def __len__(self):
        return self.Inputs.shape[0]


def load_data(batch_size):
    path_node1='/home/breeze/Desktop/SuperMeshingNet/Data/targetsData.npy'
    path_node2='/home/breeze/Desktop/SuperMeshingNet/Data/targetsData.npy'

    Inputs=np.load(path_node1)
    Targets=np.load(path_node2)

    index = [i for i in range(len(Targets))] 
    random.shuffle(index)
    Inputs = Inputs[index]
    Targets = Targets[index]

    #Set test ratio, in experiment, 0.9 is set.
    test_ratio = 0.9

    #Shuffle the dataset
    index = [i for i in range(len(Targets))] 
    random.shuffle(index)
    Inputs = Inputs[index]
    Targets = Targets[index]
    # 0-972 is train data, 972-1080 is test

    Inputs_train=Inputs[:int(len(Targets)*test_ratio),0,:,:][:,None,:,:]
    Inputs_test=Inputs[int(len(Targets)*test_ratio):,0,:,:][:,None,:,:]

    Targets_train=Targets[:int(len(Targets)*test_ratio),0,:,:][:,None,:,:] #only 1st channel kept (thinning image)
    Targets_test=Targets[int(len(Targets)*test_ratio):,0,:][:,None,:,:]   
    #Input and target is same
    Formdataset_train=FormDataset(Inputs_train,Targets_train)
    Formdataset_test=FormDataset(Inputs_test,Targets_test)

    train_loader=torch.utils.data.DataLoader(dataset=Formdataset_train,
                                        batch_size=batch_size,
                                        shuffle=True)

    test_loader=torch.utils.data.DataLoader(dataset=Formdataset_test,
                                       batch_size=batch_size,
                                       shuffle=False)
    print('Data Loading...')
    print('Inputs training dimensions:', Inputs_train.shape)
    print('Inputs testing dimensions:', Inputs_test.shape)
    print('Targets training dimensions:', Targets_train.shape)
    print('Targets testing dimensions:', Targets_test.shape)
    return train_loader, test_loader