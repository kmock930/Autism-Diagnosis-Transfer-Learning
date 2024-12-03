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

This notebook performs transfer experiments on harmful video classification using C3D [3] and R(2+1)D [4] models from Rani and Verma's work. 
It includes:
- baseline classification, 
- multi-dataset supervised contrastive learning (MSupCL) [1], 
- and self-supervised contrastive learning (SSCL) using SimCLR framework [2].

<!-- Project Background and Aims -->
## Project Background and Aims
Rani and Verma stressed the delayed diagnosis of autism spectrum disorder (ASD) done by healthcare professionals manually, 
and therefore suggested a multi-dataset supervised contrastive learning approach which automates the diagnosis in their article published in 2024 [1]

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
- Real Life Violence Situations Dataset [5] - https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset

## How to run the experiment

The experimental scripts are implemented in main.ipynb in order to facilitate the visualization of results and comparisons.

## Contact
- Tiancheng Qin: tqin021@uottawa.ca
- Kelvin Mock: kmock073@uOttawa.ca

## Acknowledgments
Some of the codes are adapted from Rani and Verma's source code [1]: https://github.com/asharani97/MDSupCL
Codes of C3D are adapted from https://github.com/hx173149/C3D-tensorflow/blob/master/c3d_model.py
Codes of R2+1D_18 are adapted from https://www.tensorflow.org/tutorials/video/video_classification

## Reference
[1] A. Rani and Y. Verma, “Activity-based early autism diagnosis using a
multi-dataset supervised contrastive learning approach,” in Proceedings
of the IEEE/CVF Winter Conference on Applications of Computer Vision
(WACV), pp. 7788–7797, January 2024.

[2] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A simple framework
for contrastive learning of visual representations,” in Proceedings of
the 37th International Conference on Machine Learning (H. D. III and
A. Singh, eds.), vol. 119 of Proceedings of Machine Learning Research,
pp. 1597–1607, PMLR, 13–18 Jul 2020.

[3] D. Tran, L. Bourdev, R. Fergus, L. Torresani, and M. Paluri, “Learning
spatiotemporal features with 3d convolutional networks,” in Proceedings
of the 2015 IEEE International Conference on Computer Vision (ICCV),
ICCV ’15, (USA), p. 4489–4497, IEEE Computer Society, 2015.

[4] D. Tran, H. Wang, L. Torresani, J. Ray, Y. LeCun, and M. Paluri, “A
closer look at spatiotemporal convolutions for action recognition,” in 2018
IEEE/CVF Conference on Computer Vision and Pattern Recognition,
pp. 6450–6459, 2018.

[5] M. M. Soliman, M. H. Kamal, M. A. El-Massih Nashed, Y. M. Mostafa,
B. S. Chawky, and D. Khattab, “Violence recognition from videos using
deep learning techniques,” in 2019 Ninth International Conference on
Intelligent Computing and Information Systems (ICICIS), pp. 80–85,
2019.
