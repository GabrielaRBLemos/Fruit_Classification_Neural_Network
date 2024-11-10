# 🍏🍐 Fruit Classification with a Neural Network

## Overview

A **deep learning-based fruit classification** system that distinguishes between **apples** and **pears** using Convolutional Neural Networks (**CNNs**) with **TensorFlow**. The model is trained on a dataset of images, where each image is categorized as either an apple or a pear.

## Dependencies

The following Python libraries are required for the project:

- TensorFlow
- NumPy
- OpenCV
- Matplotlib
- Seaborn
- Scikit-learn

You can install the dependencies using the following command:

```bash
pip install tensorflow numpy opencv-python matplotlib seaborn scikit-learn
```

## Dataset

The dataset used for this project was acquired from [Kaggle's Fruit and Vegetable Image Recognition dataset](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition). The dataset contains labeled images of various fruits and vegetables, and for this project, we focused on classifying apples and pears.

### Dataset Structure

```
- archive/
  - train/
    - apples/
    - pears/
  - test/
    - apples/
    - pears/
```

### Image Preprocessing

- Resizing: Images are resized to 64x64 pixels.
- Normalization: Pixel values are scaled between 0 and 1 to improve model convergence.

## Model Architecture

<img src="network_architecture.png" alt="Network Architecture" style="width: 12vw; height: auto;">