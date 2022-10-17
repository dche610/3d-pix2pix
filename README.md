# Repository for Project 7: Generative Deep Learning Methods for 3D Stroke MRI Lesion Inpainting

Pytorch pipeline for 3D image inpainting using Pix2Pix, with paired examples. This repository is based on:
- https://github.com/davidiommi/3D-CycleGan-Pytorch-MedImaging
- https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

*******************************************************************************
## Requirements
Packages can be installed with "pip install -r requirements.txt"
*******************************************************************************
## Important python scripts and their function

- options/base_options.py: List of base_options used to train/test the network.  

- options/train_options.py: List of specific options used to train the network.

- options/test_options.py: List of options used to test the network.

- utils: contains the Nifti_Dataset Dataloader and augmentation functions to read and augment the data, as well as data preprocessing functions. 

- models: the folder contains the scripts with the networks.

- train.py: Runs the training. (Set the base/train options first)

- test.py: It launches the inference on a folder of images chosen by the user. (Set the base/train options first)
*******************************************************************************
## Usage
### Folders structure:

Use first "organize_folder_structure.py" to create organize the data.
Modify the input parameters to select the two folders: images and labels folders with the dataset.
(Example, I want to obtain T2 from T1 brain images)


    .
	├── Data_folder                   
	|   ├── images               
	|   |   ├── image1.nii 
    |   |   ├── image2.nii 	
	|   |   └── image3.nii                     
	|   ├── labels                        
	|   |   ├── image1.nii 
    |   |   ├── image2.nii 	
	|   |   └── image3.nii  
	|   ├── masks                        
	|   |   ├── image1.nii 
    |   |   ├── image2.nii 	
	|   |   └── image3.nii  

Data structure after running it:

	.
	├── Data_folder                   
	|   ├── train              
	|   |   ├── images            
	|   |   |   ├── 0.nii              
	|   |   |   └── 1.nii                     
	|   |   └── labels            
	|   |   |   ├── 0.nii             
	|   |   |   └── 1.nii
	|   |   └── masks            
	|   |   |   ├── 0.nii             
	|   |   |   └── 1.nii
	|   ├── test              
	|   |   ├── images           
	|   |   |   ├── 0.nii              
	|   |   |   └── 1.nii                     
	|   |   └── labels            
	|   |   |   ├── 0.nii             
	|   |   |   └── 1.nii
	|   |   └── masks            
	|   |   |   ├── 0.nii             
	|   |   |   └── 1.nii
	

*******************************************************************************
### Training:
- Modify the options to set the parameters and start the training/testing on the data. Read the descriptions for each parameter.
- Afterwards launch the train.py for training. Tensorboard is not available to monitor the training: you have to stop the training to test the checkpoints weights. You can continue the training
by loading them and setting the correspondent epoch.
- To train with a second discriminator, uncomment the commented lines in file models/pix2pix_3D_model.py and make sure to comment the lines that train with a single discriminator. 
*******************************************************************************
### Inference:
Launch "test.py" to test the network. Modify the parameters in the test_options parse section to select the path of image to infer and result.

*******************************************************************************


