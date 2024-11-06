# Transfer Learning for Autism Diagnosis
## Project Background and Aims
- Rani and Verma stressed the delayed diagnosis of autism spectrum disorder (ASD) done by healthcare professionals manually, and therefore suggested a multi-dataset supervised constructive learning approach which automates the diagnosis in their article published in 2024: https://openaccess.thecvf.com/content/WACV2024/papers/Rani_Activity-Based_Early_Autism_Diagnosis_Using_a_Multi-Dataset_Supervised_Contrastive_Learning_WACV_2024_paper.pdf.
- Their model lacks training data, and thus we intend to perform a transfer learning to train their model more accurately using more abundant datasets from other domains.

## Datasets used
* TikHarm Dataset - https://www.kaggle.com/datasets/anhoangvo/tikharm-dataset
* Violence Detection - https://www.kaggle.com/datasets/yash07yadav/project-data/

## Project Steps
1. Run the notebook `data-loading.ipynb` to load the required datasets, and to get an idea about how the datasets look like. Note that those data will be loaded into a Numpy array and saved to an external file for easier development and debugging work in the future. 