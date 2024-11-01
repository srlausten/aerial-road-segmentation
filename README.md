# Aerial Road Segmentation
Segmenting roads from aerial images

## Road Segmentation using Deep Learning

This project focuses on segmenting roads from aerial images using a Fully Convolutional Network (FCN) with a ResNet50 backbone. The aim is to accurately identify road pixels in the images and distinguish them from the background.

## Project Overview

### Dataset
- Dataset from - https://www.kaggle.com/datasets/balraj98/massachusetts-roads-dataset
- The dataset consists of aerial images and corresponding binary masks indicating road (1) and background (0) pixels. The dataset is divided into training and testing sets, each containing images and their respective masks.

![Screenshot 2024-08-08 at 10 32 27 AM](https://github.com/user-attachments/assets/6d4b72ad-3a49-4486-8942-5b6d30daf1d5)

### Model Architecture
The model used for this project is the FCN-ResNet50, a Fully Convolutional Network with a ResNet50 backbone. This model is pre-trained on the COCO dataset and fine-tuned on our road segmentation dataset.

## Results 
After training the FCN-ResNet50 model for **only** 15 epochs on the road segmentation dataset, the model achieved the following performance metrics on the test dataset:

Mean IoU: 0.5127
Mean Precision: 0.7885
Mean Recall: 0.5953
Mean F1 Score: 0.6730


## Challenges
1. **Thin and Irregular Road Segments:** Roads often appear as narrow lines in aerial images, making them difficult to capture accurately.
2. **Resolution and Scale Variability:** Variability in the resolution and scale of road sections across different images affects the model's performance.


## Future Work
To improve the model's performance, the following steps are suggested:
1. **Data Augmentation:** Apply techniques such as rotation, scaling, and flipping to help the model generalize better.
2. **Training Epochs:** Train longer.
3. **Post-processing:** Use morphological operations to refine predicted road segments.
4. **Model Architecture:** Explore more advanced segmentation models like U-Net, DeepLabV3+, or models designed for thin object segmentation.
5. **Resolution Adjustment:** Train on higher resolution images or incorporate multi-scale features.

## Usage

### Prerequisites
- Python 3.6+
- PyTorch
- torchvision
- numpy
- opencv-python
- matplotlib
- scikit-learn
