# Drill bit size prediction using CNN

This repo implements a machine learning-based algorithm that can read in a png image file and identify the size of the drill bit displayed in the image.

## Tasks to be done
- Setup
    - vs code [x]
    - git [x]
- Data Generation 
    - similarity detection using phash [x]
    - save images in data_generated_images [x]
    - save data.csv in data_generated_images [x]
    - use the sample from notebook to automate this to all videos [x]
- Training script
    - what should be the training, test, validation split
    - custom dataloader for training with PyTorch
    - evaluation metrics
    - what model to use?
    - basic script that overfits and gives 0 loss
- Inference
    - script that takes input as folder and outputs filename and classified label for each image in that folder
    - also prints table of each classification found and no of images with that label