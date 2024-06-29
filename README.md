# PneumoNet: Pneumonia Detection with VGG16 Transfer Learning

This project aims to detect pneumonia from chest X-ray images using transfer learning with the VGG16 model. The project leverages advanced deep learning techniques to achieve high 
accuracy in identifying pneumonia cases.

## Folder Structure
```bash
Pneumonia-Detection/
│
├── Pneumonia_Detection.ipynb
└── README.md
```

## Project Overview

The primary goal of this project is to accurately detect pneumonia from chest X-ray images. This is achieved through the following key steps:

1.	Import Libraries: Load all the necessary libraries required for data manipulation, visualization, and model building.

2.	Define Function for Training: Create functions to handle the training process.

3.	Train Phase: Train the VGG16 model on the prepared dataset using transfer learning.

4.	Evaluate Model Performance: Assess the model's performance using various evaluation metrics.

5.	Testing Phase: Test the model on unseen data to evaluate its generalization capability.

6.	Results: Display the results and visualizations.

## Libraries Used

This project utilizes the following libraries:

•	%matplotlib inline: For displaying plots inline in Jupyter notebooks

•	copy: For creating deep copies of objects

•	matplotlib.pyplot: For plotting and data visualization

•	numpy: For numerical computations

•	pandas: For data manipulation and analysis

•	os: For interacting with the operating system

•	seaborn: For data visualization

•	skimage: For image processing

o	io, transform: For reading and transforming images

•	sklearn.metrics: For evaluation metrics (confusion matrix)

•	torch: For building and training the neural network

•	torch.nn: For defining neural network layers

•	torch.optim: For optimization algorithms

•	torchvision: For image transformations and datasets

o	datasets, models, transforms: For dataset handling, pre-trained models, and image transformations

## Getting Started
To get started with the project, follow these steps:

1.	Clone the repository:
```bash
git clone https://github.com/yourusername/Pneumonia-Detection.
git cd Pneumonia-Detection
```
2.	Install the required libraries:
```bash
pip install -r requirements.txt
```
3.	Access the dataset and pre-trained models from Kaggle:
o	Dataset and Models on [Kaggle🔗](https://www.kaggle.com/code/dhakshanamoorthyr/pneumonet-pneumonia-detection-using-vgg16)

4.	Run the Jupyter Notebook to execute the entire workflow:
```bash
jupyter notebook "Pneumonia_Detection.ipynb"
```

## Project Workflow

1.	Import Libraries: Import all the necessary libraries for the project.

```bash 
%matplotlib inline
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
import skimage
from skimage import io, transform
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
```

2.	Define Function for Training: Define functions to handle the training process, including loss calculation and backpropagation.

3.	Train Phase: Train the VGG16 model on the dataset using transfer learning. Fine-tune the pre-trained model on the pneumonia dataset.

4.	Evaluate Model Performance: Use metrics like confusion matrix to evaluate the model's performance on the validation set.

5.	Testing Phase: Test the model on unseen data to assess its generalization capability.

6.	Results: Display the results, including accuracy, loss curves, and confusion matrix.

![image](https://github.com/DhakshanaMoorthyRDM/LightGBM_Gold_Forecast/assets/121345776/248ad69d-a6ba-460a-a9d2-ee8807c3ec31)


## Acknowledgements

•	The developers and maintainers of the libraries used in this project.

•	The contributors to the chest X-ray dataset.

•	Kaggle for providing the dataset and pre-trained models.

Feel free to explore, contribute, and provide feedback!

