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

This file, `README.md`, should be used to include all information on the files included in the project and how to make them work.

It makes use of Markdown, which is a markup language focused on readability. You can find information on the main syntax of Markdown in:

[Markdown Cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)

[Markdown Basics Guide](https://markdown-guide.readthedocs.io/en/latest/basics.html)

Multiple text editors, such as Visual Studio Code, Sublime Text, Atom, Notepad++ have extensions that you can install to transform the Markdown file (.md) into a .pdf or .html.
