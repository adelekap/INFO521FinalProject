# Dissociating Performance Tiers on Memory Tasks Across the Lifespan of the Rat: An evaluation of various machine learning models to analyze cognitive aptitude

To install all required packages:
`pip install -r requirements.txt`



Data Files
--------------


`CASWatermaze.csv`
This csv has the water maze task data.


`CASWorkingMemory.csv`
This csv has the working memory task data.


`data.py
This module houses the dataframes required for training and testing models.



Python Scripts
_________________


`KMeans.py` 
Run this to perform performance clustering using KMeans


`KNN.py` 
Run this to see prediction of ages using KNN


`lasso.py`
Run this to fit lasso linear model using performances of both tasks 


`learning.py` 
Fits linear models of different polynomial degrees to the water maze data


`LeastSquares.py` 
Fits and visualizes linear models for 1 and 2 features


`naiveBayes.py` 
Fits data for gaussian, multinomial, and bernoulli naive bayes


`SVM.py` 
Trains SVM classifier and plots contours
