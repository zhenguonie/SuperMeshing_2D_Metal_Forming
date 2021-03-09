import time
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import random
import Dataset.dataloader as dataloader
from Model.SuperMeshingNet import SuperMeshingNet
import torch
import matplotlib.pyplot as plt

num_channel = np.array([16,32,64,128,256,512])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 20

model = SuperMeshingNet(num_channel,batch_size)
model = model.to(device)
train_loader, test_loader = dataloader.load_data(batch_size)

model.load_state_dict(torch.load('SuperMeshingNet_600Epoch.pkl',map_location=torch.device('cpu')))
maxInputMinusOutput = []
mse_all = []
mae_all = []
maeloss = nn.L1Loss()
mseloss = nn.MSELoss()
sum_mae_train = 0
sum_mae_test = 0
sum_mse_train = 0
sum_mse_test = 0
for batch_idx, (features, targets) in enumerate(train_loader):
    features = features.to(device)
    decoded, encoded1 = model(features)
    targets = targets.to(device)
    sum_mae_train += float(maeloss(targets,decoded))
    sum_mse_train += float(mseloss(targets,decoded))
for batch_idx, (features, targets) in enumerate(test_loader):
    features = features.to(device)
    decoded, encoded1 = model(features)
    targets = targets.to(device)
    sum_mae_test += float(maeloss(targets,decoded))
    sum_mse_test += float(mseloss(targets,decoded))
    for i in range(0,batch_size):
        
        a=targets[i].view(256,256)
        
        image = a.cpu().clone()
        image = image.squeeze(0)
        d=decoded[i].cpu().detach().numpy().reshape(256,256)
        

        maxInputMinusOutput.append(abs(float(image.max()-d.max())))
        
        mse_all.append(float(mseloss(targets[i],decoded[i])))
        mae_all.append(float(maeloss(targets[i],decoded[i])))
mae_train = sum_mae_train/81
mae_test = sum_mae_test/9
mse_train = sum_mse_train/81
mse_test = sum_mse_test/9

print("the mse train is: ", mse_train)
print("the mse test is: ", mse_test)
print("the mae train is: ", mae_train)
print("the mae test is: ", mae_test)

statistic = {'1E-06':0,'1.5E-06':0,'2E-06':0,'2.5E-06':0,'3E-06':0,'4E-06':0,'5E-06':0,'1E-05':0}
for i in mse_all[0:180]:
    if abs(i)<0.000001:
        statistic['1E-06']+=1
    elif abs(i)<0.0000015:
        statistic['1.5E-06']+=1
    elif abs(i)<0.000002:
        statistic['2E-06']+=1
    elif abs(i)<0.0000025:
        statistic['2.5E-06']+=1
    elif abs(i)<0.000003:
        statistic['3E-06']+=1
    elif abs(i)<0.000004:
        statistic['4E-06']+=1
    elif abs(i)<0.000005:
        statistic['5E-06']+=1
    elif abs(i)<0.00001:
        statistic['1E-05']+=1
    
    else:
        statistic['1E-05']+=1
s_name = []
s_value = []    
for i in statistic:
    s_value.append(statistic[i])
    s_name.append(i)
plt.bar(s_name,s_value)