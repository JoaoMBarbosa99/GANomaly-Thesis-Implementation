# Advanced Automation Project Structure

*A logical, reasonably standardized, but flexible project structure for doing and sharing work. (based on [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/))*

This repository contains the base structure for Advanced Automation project. In a near future, your thesis can also use this structure to make your work available for contributions from other authors. 

Please follow the folder structure presented as close as you can. As all projects have different requirements, you will probably have to change the structure a bit to fit your needs.

### Folder structure

The directory structure of your new project looks like this (please adjust the structure and its description to best fit your project): 

```
├── README.md          <- The top-level README for contributers of this project.
│
├── data
│   ├── raw            <- The original, immutable data dump.
│   ├── interim        <- Intermediate data that has been transformed.
│   └── processed      <- The final, canonical data sets for modeling.
│
├── docs               <- Documents
│   ├── reports        <- Generated analysis as HTML, PDF, LaTeX, DOCX etc.
│   ├── figures        <- Generated graphics and figures to be used in reporting
│   └── project/thesis <- Project/Thesis source files
│       └── templates  <- Templates (if available)
│
├── references         <- Articles, books and other references used in your project. It is good practice to  
|                         reference these materials in the comments of your code.
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering), the creator's  
|                         initials, and a short `-` delimited description, e.g. `1.0-aa-initial-data-exploration`.
│
├── requirements.txt   <- The file with instructions for reproducing the project environment (software).  
|                         Indicates the version of  software you are using. If using Python, you can do 
|                         `pip freeze > requirements.txt`  to generate a list with the version of the 
|                         different packages you use
│
├── code               <- Source code for use in this project.
|    ├── data          <- Scripts to download, generate and process data
|    │   └── make_dataset.py/m
|    │   └── process_dataset.py/m
|    │
|    ├── algorithms    <- Functions to create models, run models, optization algorithms, etc.
|    │
|    ├── results       <- Scripts to apply the algorithms and obtain the results of your project
|    │
|    └── visualization <- Scripts and functions for visualizations
|
└── laboratory         <- Laboratory classes (add only the necessary files)
     ├── L1_Git        <- Github
     ├── L2_GCollab    <- Cloud Computing: Google Collab
     ├── L3_DB         <- Databases and SQL Query
     └── L4_Arduino    <- Hardware implementation: Arduino
```


### Readme file

This file, `README.md`, should be used to include all information on the files included in the project and how to make them work.

It makes use of Markdown, which is a markup language focused on readability. You can find information on the main syntax of Markdown in:

[Markdown Cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)

[Markdown Basics Guide](https://markdown-guide.readthedocs.io/en/latest/basics.html)

Multiple text editors, such as Visual Studio Code, Sublime Text, Atom, Notepad++ have extensions that you can install to transform the Markdown file (.md) into a .pdf or .html.
