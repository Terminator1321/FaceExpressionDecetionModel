#NOTE : I haven't properly trained the model yet its accuracy is around 44% so trained it first 

# FaceExpressionDecetionModel
This repository contains a simple AI model for facial emotion detection, capable of recognizing seven emotions: anger, sadness, happiness, neutral, disgust, fear, and surprise. It includes the trained model, Face_ExpressionAI_train.ipynb for training, ModelTest.py for testing, and datasets for training and testing the model.

## Overview
This project demonstrates a deep learning-based approach to facial emotion detection. Using a CNN model, it analyzes facial expressions to classify emotions into 7 categories.


## Features
- Detects 7 emotions: anger, sadness, happiness, neutral, disgust, fear, and surprise.
- Simple and lightweight AI model.
- Ready-to-use with pre-trained model files.
- Easily customizable with your own datasets.


## Project Structure
├── face_classifier/           
├── model/                     
├── ModelTest.py              
├── train_data/                
├── test_data/                 
└── Face_ExpressionAI_train.ipynb

## Installation
1. Clone this repository:
   git clone <repository_url>

2. Install dependencies:
   pip install -r requirements.txt

## Usage
- Train the model:
  Open Face_ExpressionAI_train.ipynb and run all cells.
- Test the model:
  python ModelTest.py

## 📦 Dataset
The training and testing data used in this project were sourced from kaggle ([https://www.kaggle.com/...](https://www.kaggle.com/datasets/msambare/fer2013/data)) 
