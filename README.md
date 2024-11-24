<!-- Adapted from https://github.com/othneildrew/Best-README-Template/blob/main/README.md -->
<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h3 align="center">Harmful Video Classification By Using Contrastive Learning</h3>
  <p align="center">
    This Project performs experiments on harmful video classification using C3D and R(2+1)D models. 
    <br />
  </p>
</div>


<!-- ABOUT THE PROJECT -->
## About The Project

This notebook performs transfer experiments on harmful video classification using C3D and R(2+1)D models from Rani and Verma's work. 
It includes:
- baseline classification, 
- multi-dataset supervised contrastive learning (MSupCL), 
- and self-supervised contrastive learning (SSCL) using SimCLR framework.

<!-- Project Background and Aims -->
## Project Background and Aims
Rani and Verma stressed the delayed diagnosis of autism spectrum disorder (ASD) done by healthcare professionals manually, 
and therefore suggested a multi-dataset supervised contrastive learning approach which automates the diagnosis in their article published in 2024: https://openaccess.thecvf.com/content/WACV2024/papers/Rani_Activity-Based_Early_Autism_Diagnosis_Using_a_Multi-Dataset_Supervised_Contrastive_Learning_WACV_2024_paper.pdf.

The objective of this project is to explore whether the MSupCL method proposed by Rani and Verma has performance enhancement for harmful video classification.

<!-- GETTING STARTED -->

## Prerequisites

This project was built using Python 3.9.20, CUDA 11.2, cuDNN 8.1.

Please start by installing the dependencies for this project. Run the following code:

  ```sh
  pip install -r requirements.txt
  ```

## Datasets Used:
- TikHarm Dataset - https://www.kaggle.com/datasets/anhoangvo/tikharm-dataset
- Real Life Violence Situations Dataset - https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset

## How to run the experiment

The experimental scripts are implemented in main.ipynb in order to facilitate the visualization of results and comparisons.

## Contact
- Tiancheng Qin: tqin021@uottawa.ca
- Kelvin Mock: kmock073@uOttawa.ca

## Acknowledgments
Some of the codes are adapted from Rani and Verma's source code: https://github.com/asharani97/MDSupCL
