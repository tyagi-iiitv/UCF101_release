## Action Recognition @ UCF101  

### 1. Video Images  

UCF101 dataset contains 101 actions and 13,320 videos in total.  

+ `annos/actions.txt`  
  + lists all the actions (`ApplyEyeMakeup`, .., `YoYo`)   
  
+ `annots/videos_labels_subsets.txt`  
  + lists all the videos (`v_000001`, .., `v_013320`)  
  + labels (`1`, .., `101`)  
  + subsets (`1` for train, `2` for test)  

+ `images_class1/`  
  + contains videos belonging to class 1 (`ApplyEyeMakeup`)  
  + each video folder contains 25 frames  


### 2. Video Features

+ `extract_vgg16_relu6.py`  
  + used to extract video features  
     + Given an image (size: 256x340), we get 5 crops (size: 224x224) at the image center and four corners. The `vgg16-relu6` features are extracted for all 5 crops and subsequently averaged to form a single feature vector (size: 4096).  
     + Given a video, we process its 25 images seuqentially. In the end, each video is represented as a feature sequence (size: 4096 x 25).  
  + written in PyTorch; supports both CPU and GPU.  

+ `vgg16_relu6/`  
   + contains all the video features, EXCEPT those belonging to class 1 (`ApplyEyeMakeup`)  
   + you need to run script `extract_vgg16_relu6.py` to complete the feature extracting process  
