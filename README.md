# Ganomaly Thesis Implementation

*A logical, reasonably standardized, but flexible project structure for doing and sharing work. (based on [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/))*

This repository contains the base structure for the implementation used in my thesis containing the adjacent scripts developed.

### Folder structure

The directory structure of your new project looks like this (please adjust the structure and its description to best fit your project): 

```
├── README.md          <- README file.
│
├── data
│   ├── train          <- Train data (only normal images).
│       └── 0.normal
│   ├── test           <- Test data used for validation.
│       └── 0.normal
│       └── 1.abnormal
│
├── lib                <- Ganomaly files
│   ├── visualizer.py        <- 
│   ├── data.py              <- 
│   ├── __init__.py          <-
│   ├── networks.py          <-
│   ├── evaluate.py          <-
│   ├── loss.py              <-
│   └── model.py             <- Project/Thesis source files
│
├── options.py         <-
├── train.py           <-
├── metrics.py         <-
│ 
├── references         <- Articles, books and other references used in your project. It is good practice to  
|                         reference these materials in the comments of your code.
│
├── requirements.txt   <- The file with instructions for reproducing the project environment (software).  
|                         Indicates the version of  software you are using. If using Python, you can do 
|                         `pip freeze > requirements.txt`  to generate a list with the version of the 
|                         different packages you use
│
├── output         <- Laboratory classes (add only the necessary files)
     ├── test_results.csv        <- test results in csv file for (result of metrics.py)
     ├── test_results.xlsx       <- test results in xlsx file for (result of metrics.py)
     ├── train_results.csv       <- train results in csv file for (result of metrics.py)
     └── train_results.xlsx      <- train results in xlsx file for (result of metrics.py)
│
├── Sampling.py         <- Script for sampling data
|
└── RLHF
     ├── CustomEnv.py
     └── HumanOperator.py 
```


### Readme file

## Implementation

# 1. Import Requirements
     import sys
     import os
     from os import system
     import tensorflow as tf
     from google.colab import drive (only if you are using Google Colab)
     
# 2. Change directory

If you use the Google Colab notebook, connect it to your drive using: 
     drive.mount('/content/drive')
     os.chdir('/content/drive/MyDrive/path') - change"path" to your desired path

If you are using other notebook such as Jupyter Notebook just change its director using OS.

# 3. Train the model

To train the GANomaly model just use the following text where some of the hyerparameter have been changed:

!python train.py                             \
        --model ganomaly                     \
        --metric auprc --dataroot data --niter 100 \
        --lr 0.0004 --beta1 0.9 --isize 16 --proportion 0.5 \
        --batchsize 128 --nz 250  --w_con 50  --w_enc 1  --w_adv 1
        
# 4. Metrics.py

After the previous text (# 3) is ran, 4 folders are created in the output folder.
For a complete analysis of run:

!python metrics.py

This will provide various graphs for the different epochs previously ran and a new folder containing its respective images is created.


## Sampling.py (for SIXRAY only)

This code serves to randomly generate 3 new folders (train/0.normal; test/0.normal; test/0.abnormal) containing the desired number of images on each. Just make sure to chang the directory on the script itself to your local sixray dataset.

## RLHF Files

The 2 present scripts display some of the developments for Reiforcement Learning w/ Human Feedback to be implemented in the future.
