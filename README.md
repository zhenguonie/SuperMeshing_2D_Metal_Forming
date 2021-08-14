
# 2021_SuperMeshing_2D_Metal_Forming

This is the implementation of the SuperMeshingNet, for paper: SuperMehsingNet: A New Architecture for Stress Field Mesh Density Increase of Metal Forming with Attention Mechanism and Perceptual Features

The structure of the SuperMeshingNet is shown in Figure.The main structure of the model is composed of ResNet with thechannel and spatial attention modules, and the down-samplinglayer and the up-sampling layer are connected by a structurethat skip-connection.  After four down-sampling modules, we added the Res34 structureto deepen the network and improve the effect. 
Meanwhile,a geometric attention map that contains the information of ge-ometric feature generated from the training set is learned by ageometric extractor and added on the last third upsampling mod-ule to highlight the attention of the model on the image of theFEA result. 
Moreover, a perceptual feature extractor composedby ResNet is implemented to optimize the model by perceptualloss, which enhancing the performance in feature space.
<div align=center><img width="620" height="320" src="https://user-images.githubusercontent.com/53844777/110433887-089bd680-80ec-11eb-8907-7b676db8e33b.png"/></div>



To train the model, please download the project and run:

python train.py

To evaluate the model, please run:

python test.py

If you want to test different scaling of factors, please change the data processing method in /Dataset/Dataloader.py, replacing 2X as 4X or 8X.

The picture floder includes the reslt of experiment.
<div align=center><img width="510" height="560" src="https://user-images.githubusercontent.com/53844777/110434544-eb1b3c80-80ec-11eb-9bcd-3f98ff05a924.png"/></div>

# Data
You can download the data : https://drive.google.com/drive/folders/1FlxDkLyrJYCpSIcXii7AiYZDkXLUk3jD?usp=sharing
