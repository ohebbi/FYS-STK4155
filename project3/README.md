# Solving partial differential equations & Eigenvalue Problems with neural networks
This project is divided into two parts where the first part is about solving partial differential equations using Forward Euler, Backward Euler, Crank-Nicolsen and a Neural Network. The second part is about finding eigenvalues using a neural network.


## Code skeleton
- functions
  - NN_Eigenvalue_solver.py (this is where all of the functions are)
  - PDE_solver.py
- plots
  - PDE
    - .. (PGFs for the report) 
  - eigenvalue_solver
    - .. (PGFs for the report)
- report
  - project3_report.pdf
- main.py 
- nn_eigenvalue.ipynb
- nn_eigenvalue_random.ipynb
- solving_PDE.ipynb

## Jupyter notebooks
Since this project is divided into two parts, there are two jupyter notebooks which gives you an overview and our results (with a standard deviation and random samples that changes every run), and how we got them. The name of them are 
- solving_PDE.ipynb. This file finds solution to the PDEs.
- nn_eigenvalue.ipynb. This file finds eigenvalues for given matrices using neural network.
- nn_eigenvalue_random.ipynb This file finds eigenvalues for a random given matrix using neural network. 


## How to run the files 
The required packages included to run the jupyter notebooks are 
- numpy
- matplotlib
- mpl_toolkits
- sklearn
- tqdm
- jupyter notebook
- seaborn

Fork this repository, enter the jupyter notebooks, and run all the cells in one run. You can also run main.py, where there is small summary of the jupyter notebooks. 
