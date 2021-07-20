This folder contains the functions found in the paper An Evolutionary Approach to Feature Selection and Network Optimisation.

There are four python files:

Main.py - This is the main file that calls the other functions and contains the loops for running the genetic algorithms and collecting data
Preprocessing.py - This file contains the process for preprocessing the raw data. The raw data must be in the working directory and after running
this file a csv of processed data and a separate csv of labels will be written into the working directory.
EA_funcs.py - This file contains the functions that describe the genetic algorithm. This includes the creation of chromosomes, evaluation with
an SVM classifier, selection, crossover, mutation and so forth. These functions are called by Main.py
nn_funcs.py  - This file contains the two functions related to the artificial neural network classifier which is used in the optimization algorithm.


These files were written on a computer running windows 10 and python 3.8.8


