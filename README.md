# imageSeg

Important note: this work is still not complete ...

2D and 3D image segmentation for binary (semantic segmentation) or multi-class (instance segmentation) using Deep Learning.

This is a useful resource for image segmentation. It contains collection of different PyTorch and Tensorflow implementation with their video explaination in Youtube.  

The discuession section is helpful to exchange information and ask questions. 

![U-Net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png "U-Net Structure")


## Contents:

Image Segmentation Using U-Net

  - Paper: [2015, Ronneberger et al, U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
  - Perssons: 2D implementation and explaination by Aladdin Persson.
    - Paper explaination: [video](https://youtu.be/oLvmLJkmXuc?si=k-aJ1UtrEr8qu-hj)
    - Paper implementation (pytorch): [video](https://youtu.be/IHq1t7NxS8k?si=cd-9x6pnHLFMCdgg)    
    - Carvana Dataset: [carvana-image-masking-challenge](https://www.kaggle.com/c/carvana-image-masking-challenge)
    - Perssons Code: [Semantic Segmentation U-Net](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet)
    - To run:

           # requiremnts: pip install albumentations tqdm  
           # The file contains calls for both Pytorch and Tensorflow, you can run one of them
           # One should get over 90% dice after one iteration
           # Modify the parameters in the file as needed then     
           python3 Persson/demoUnet.py 

  - DigitalSreeni: 2D implementation and explaination by DigitalSreeni.
    - Paper explaination: [video](https://youtu.be/azM57JuQpQI?si=bHNzo8a-NFLbXRn1)
    - Paper implementation videos: [1](https://youtu.be/azM57JuQpQI?si=d85pKOlDoJPcasmF),[2](https://youtu.be/68HR_eyzk00?si=ND08rdEAWQzf9lM2),[3](https://www.youtube.com/watch?v=sb0uglcqO2Y&list=PLZsOBAyNTZwbR08R959iCvYT3qzhxvGOE&index=3),[4](https://www.youtube.com/watch?v=0kiroPnV1tM&list=PLZsOBAyNTZwbR08R959iCvYT3qzhxvGOE&index=4),[5](https://www.youtube.com/watch?v=cUHPL_dk17E&list=PLZsOBAyNTZwbR08R959iCvYT3qzhxvGOE&index=5),[6](https://www.youtube.com/watch?v=RaswBvMnFxk&list=PLZsOBAyNTZwbR08R959iCvYT3qzhxvGOE&index=6),[7](https://www.youtube.com/watch?v=J_XSd_u_Yew&list=PLZsOBAyNTZwbR08R959iCvYT3qzhxvGOE&index=8)
    - Dataset: 
    - DigitalSreeni Code (keras): [Segmentation U-Net](https://github.com/bnsreenu/python_for_microscopists/blob/master/074-Defining%20U-net%20in%20Python%20using%20Keras.py)
    - To run:

           # Modify the parameters in the file as needed then     
           python3 DigitalSreeni/demoUnet.py 

### TODOS:

  - Complete functionality
    - [ ] Add complete dataset pre-processing steps    
    - [ ] iaUtils/aug: Augmentation
        - [ ] use opencv 
        - [ ] add support to offline and online augmentaion
    - [ ] Add prediction for one or multiple images
    - [ ] Dataset: Use two folders approach
        -data: # from this we create train, validation, and test
        - img
        - seg
  - [ ] Test on datasets for binary and semantic segmentation e.g. spine cochlea 
  - [ ] Add DigitalSreeni implementation
  - [ ] Add the kaggle Carvana winner implementation
        - [Code for Dice: 0.99733](https://github.com/asanakoy/kaggle_carvana_segmentation)
  - [ ] Impement 3D Model
  - [ ] Add sample results to the readme file 
        - images, plots 
  - [ ] Add pre-trained models and sample demo files
- [ ] Add [Segment Anything Model (SAM) implementation](https://www.youtube.com/watch?v=83tnWs_YBRQ)
