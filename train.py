import time
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import random
import Dataset.dataloader as dataloader
from Model.SuperMeshingNet import SuperMeshingNet
import torch

num_epochs= 1000
learning_rate = 0.0005
batch_size = 20
num_channel = np.array([16,32,64,128,256,512])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

random_seed = 37
torch.manual_seed(random_seed)
model = SuperMeshingNet(num_channel,batch_size)
model = model.to(device)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_loader, test_loader = dataloader.load_data(batch_size)
print('Initialize training SuperMeshingNet...')
KLLoss = nn.KLDivLoss()
L1Loss = nn.L1Loss()
L2Loss = nn.MSELoss()
start_time = time.time()
print('Start training SuperMeshingNet...')
for epoch in range(num_epochs):

    for batch_idx, (features, targets) in enumerate(train_loader):
        features = features.to(device)
        targets = targets.to(device)

        #Use the SuperMeshingNet as the perceptual feature extractor
        d_useless, perceptual_real = model(targets)
        decoded, perceptual = model(features)
        
        #Loss function
        cost_content = L1Loss(decoded, targets)
        cost_geoAttention = KLLoss(decoded, targets)
        cost_perceptual = L2Loss(perceptual_real, perceptual)
        
        cost = 0.8*cost_content + 0.1*cost_geoAttention + 0.1*cost_perceptual;
        
        optimizer.zero_grad()
        del features
        torch.cuda.empty_cache()
        del targets
        torch.cuda.empty_cache()
        del decoded
        torch.cuda.empty_cache()
        cost.backward()
        
        ### Update model parameters
        optimizer.step()
        
        ### Logging
        if not batch_idx % 50:
            print ('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.8f' 
                   %(epoch+1, num_epochs, batch_idx, 
                     len(train_loader), cost))
            

            
    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
    #Saving model
    if epoch == 200:
        torch.save(model.state_dict(), 'SuperMeshingNet_200Epoch.pkl')
    if epoch == 400:
        torch.save(model.state_dict(), 'SuperMeshingNet_400Epoch.pkl')
    if epoch == 600:
        torch.save(model.state_dict(), 'SuperMeshingNet_600Epoch.pkl')
    if epoch == 800:
        torch.save(model.state_dict(), 'SuperMeshingNet_800Epoch.pkl')
    if epoch == 1000:
        torch.save(model.state_dict(), 'SuperMeshingNet_1000Epoch.pkl') 
    
print('Finishing Training. Total Training Time: %.2f min' % ((time.time() - start_time)/60))