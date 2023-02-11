# Drill bit size prediction using CNN

This repo implements a machine learning-based algorithm that can read in a png image file and identify the size of the drill bit displayed in the image.

## Running this code locally
I have provided the `requirements.txt` which contains the list of libraries necessary to run the code. It is recommended to have a virtual environment set up (venv or conda) before running 
```
pip install -r requirements.txt
```

Please ensure the videos are in the folder `data_raw_videos`

**Data Generation**:
```
$ python .\src\generate_training_data.py
```
This script will take the videos form folder `data_raw_videos`.  
Output:
 - training images in the `data_generated_images` folder
 - `data.csv` file in the `data_generated_images` folder

**Training**:
```
$ python .\src\train.py
```
Run this script will train the model. (image below shows screenshot of this)  
Output:
- `model.pth` in `models` folder. 
- `test.csv` file in `data_generated_images` folder

**Inference on a folder**  
This script will do inference on all the images presend in the folder that is passed as an argument to this script. (image below shows screenshot of this)
```
$ python .\src\folder_inference.py <folder_name>
```

## Approach
- I have made the simplifying assumption that the drill bit will always be in the same part of the image. If this doesn't hold, then we can go for an approach that would do the object localization first
- Thus, I created the training data by cropping a small 140x40 sized image from the captured frames
- I used perceptual hashing to filter out frames if they are very similar to an existing frame. Reason for doing this is if we have very similar frames, then our model could suffer from leakage between test and train data.
- A total of 571 images were generated ~ 30 images per video
- I modeled the problem as a multiclass regression problem as the response variables (width and height) is continous in nature. Since I am regressing on the width and the height, classes are inferred from the predicted output
- I have gone for a simple CNN architecture with only 1 convolutional layer followed by maxpooling and a fully connected layer. There is definitely a scope to optimize the model architecture but I have avoided doing so due to the time constraint
- I have used data augmentation during training to avoid overfitting.
- Image below shows the confusion matrix after early stopping on the train, val set. Train:val:test split was 60:20:20 for this phase
- Model was finally retrained using a 70:30 train:val split

Confusion matrix:
![confusion matrix](/images/confusion_matrix_test.png)

Training screenshot:
![training](/images/training.png)

Inference screenshot:
![inference](/images/inference.png)

## Next steps  

To build up on this, I would
- do error analysis to identify the examples where the model is making more errors
- experiment with different model architectures