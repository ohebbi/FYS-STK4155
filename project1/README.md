# Model of Franke's function and a geographic area in Norway using five-fold cross validation and three regression types. 
This project is divided into two parts. The first part is modeling Franke's function, while the second part is modeling some given terrain data from Norway. 

## Code skeleton
- datafiles
  - .. (.tif files which is our model for the terrain)
- functions
  - functions.py (this is where all of the functions are)
- plots
  - Franke
    - .. (PDFs) 
  - Terrain
    - .. (PDFs)
- report
  - project1_report.pdf
- main.py 
- main_Franke.py
- main_terrain.py

## Jupyter notebooks
Since this project is divided into two parts, there are two jupyter notebooks which gives you an overview and our results (with a standard deviation and random samples that changes every run), and how we got them. The name of them are 
- main_Franke.py. This file models Franke's Function.
- main_terrain.py. This file models the terrain data.


## How to run the files 
The required packages included to run the jupyter notebooks are 
- numpy
- matplotlib
- imageio
- sklearn
- mpl_toolkits
- tqdm
- jupyter notebook
- seaborn
- pandas

Do a git pull, enter the jupyter notebooks, and run all the cells in one run. You can also run main.py, where there is small summary of the jupyter notebooks. 
