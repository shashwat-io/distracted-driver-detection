# 🚗 Distracted Driver Detection using Deep Learning

## 📌 Overview
This project implements a deep learning–based image classification system to detect distracted driving behaviors using in-car driver images. The model classifies each image into one of 10 driver activity categories.

The solution is built using **transfer learning with ResNet50**, trained and evaluated on the **State Farm Distracted Driver Detection** dataset from Kaggle.

---

## 🧠 Problem Statement
Given an image of a driver inside a car, predict the driver’s activity such as:
- Safe driving
- Texting (left/right)
- Talking on phone (left/right)
- Operating radio
- Drinking
- Reaching behind
- Hair & makeup
- Talking to passenger

---

## 🛠️ Tech Stack
- Python
- PyTorch
- TorchVision
- NumPy, Pandas
- scikit-learn
- Kaggle GPU (NVIDIA T4)
- Transfer Learning (ImageNet pretrained ResNet50)

---

## 🏗️ Approach
1. Loaded and explored the Kaggle dataset
2. Applied image preprocessing and data augmentation
3. Used ResNet50 pretrained on ImageNet
4. Replaced the final classification layer for 10 classes
5. Trained using AdamW optimizer and cosine learning rate scheduling
6. Evaluated performance using validation accuracy and log loss
7. Generated probability-based Kaggle submission (`c0`–`c9`)

---

## 📊 Results
- Validation Accuracy: **~95%**
- Kaggle Log Loss: **~0.39**
- Model shows strong generalization on unseen test data

---

## 📂 Dataset
**State Farm Distracted Driver Detection (Kaggle)**  
https://www.kaggle.com/c/state-farm-distracted-driver-detection  

⚠️ Dataset is not included in this repository due to Kaggle licensing.

---

## 🚀 How to Run
Open the notebook:

distracted_driver_detection_resnet50.ipynb


Run all cells in order.  
GPU acceleration is recommended.

---

## 👤 Author
**Shashwat Kumar**  
GitHub: https://github.com/shashwat-io
