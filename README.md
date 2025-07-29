# Towards Digital Twin of RF Ablation: Real-Time Prediction of Time-Dependent Thermal Effects Using Transformer

## Overview
This repository contains the implementation of the neural network model presented in the MICCAI 2025 workshop [Digital Twin for Healthcare], as described in the paper "Towards Digital Twin of RF Ablation: Real-Time Prediction of Time-Dependent Thermal Effects Using Transformer."

## Features
- A neural network model for prediction of RF ablation outcomes(damage area and temperature distribution) based on [UNETR: Transformers for 3D Medical Image Segmentation] by Ali Hatamizadeh, Dong Yang, Holger Roth, and Daguang Xu (2021).
- The damage prediction model uses Dice loss, while the temperature prediction model uses MSE loss. Both models share the same UNETR architecture, but their parameters are trained separately.

## Requirements
- python 3.10.12
- cuda 11.8
- torch 2.0.1
- numpy 1.26.4

## Dataset
- The composition of the dataset
    - input : a 3D tumor image and a 3D needle image
    - output : an 18-channel 3D image(damage area or temperature distribution) over time.
-  The tumor image was constructed using the publicly available dataset [Saha et al., 2018](https://www.nature.com/articles/s41416-018-0185-8).

## References
- 'UNETR.py' was designed with reference to [UNETR implementation by tamasino52](https://github.com/tamasino52/UNETR), which is licensed under the MIT License.

## Contact
For any queries, please reach out to Seonaeng Cho. (seonaeng@yonsei.ac.kr)
