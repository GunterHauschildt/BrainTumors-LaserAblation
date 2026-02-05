# BrainTumors-LaserAblation

Preliminary investigation into datasets and techniques for machine learning with respect to laser ablation.  
This project's goal: Using un-annotated images, find the optimal path for a laser ablation of a tumor.  
   
### Brain Structures   
<img width="847" height="732" alt="structures" src="https://github.com/user-attachments/assets/da17a92b-64ba-4a46-b102-73160338b00b" />    
      
### Tumor   
<img width="842" height="737" alt="brats_tumor" src="https://github.com/user-attachments/assets/b1dc91ff-c758-41e6-bd57-86e4d4bb8acc" />    
        
### Best Laser Path   
<img width="857" height="734" alt="laser" src="https://github.com/user-attachments/assets/8101ed47-ff55-4f19-8537-eb2d99772df6" />    

### 2D Prototype
A much simpler, 2D blob-based, self-supervised-learning model was trialed. We can see that a simple CNN can be used to find the
laser path & width (blue) that finds the tumor (red) while avoiding important structures (green).

https://github.com/user-attachments/assets/5fce49ed-5927-40de-bc94-5bc3d4ee98aa


## Datasets & Segmentations tools
Common segmented-tumor datasets are those used for the Brats competitions. The 2020 dataset contains non-nifti data, raw slices, making
integration into a nifti-based pipeline error prone. The 2021 dataset, while nifti, is heavily preprocessed, including skull-stripping and face removal.  
The 2021 dataset has T1, T1ce, T2, Flair images and the tumors are annotated in a segmentation mask.  2022 on newer haven't been investigated (yet).  
  
IXI datasets contain raw MRI.
  
Instead of annotated brain structure datasets, FreeSurfer (by its NN  GPU enhanced cousin FastSurfer) can be used to segment brain structures.
Being atlas based, FreeSurfer won't work well with skull-stripped/face-removed images with large tumors: so Brats2021 is out.
FreeSurfer does work well on the IXI dataset.  
Instead, tumors from the Brats dataset are extracted and placed as anomalies in the IXI image.  
Proceeduraly speaking, these anomalies are treated as a MonAI transform and randomly generated in the training pipeline.  
FreeSurfer creates a lot of brain structure labels: if one-hot'ed, the memory usage would be out of reach. If not, the labels very close to each other during training.  
Instead the FreeSurfer labels are relabelled into a 10 like-function groups. See Transforms\RelabelFreeSurfer\free_surfer_relabel.json.  
  
## Results  
### Segmentation  
The base CNN is Unet with 16/32/64/128/256 filters with two ResNet blocks per layer. Relatively speaking thats pretty small, but also pretty big for my RTX GPU.  
Standalone segmentation of the Brats dataset gives a Dice score of about .85. Thats pretty good: so its expected that we should be able to extract tumors.  
Training of the 10 functional groups on FreeSurfer segmented images gives a Dice score of about .65.  
A radiologist's opinion is unavaible to to see if the tumor-extraction-insertion algorithm gives realistive results. It is likely further work is required.    
Training of the FreeSurfer with tumors gives a Dice score if about .62.  
### Laser Ablation  
Likewise, a simple linear regression 3D CNN is used to calculate the laser path. No metrics are available to determine its performance, but eyeballing, it looks close.
With no ground truth, self-supervised learning is used.
