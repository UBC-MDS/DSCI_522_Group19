# Wine Quality Score Predictor

-   Contributors: Kingslin Lv, Manju Neervaram Abhinandana Kumar, Zack Tang, Paval Levchenko

Wine Quality Score Predictor is our Data analysis project for DSCI 522.

## Introduction

  In this project, we aim to predict the wine taste prefrences as quality scores ranging from 0 to 10 based on physicochemical properties of the wines and sensory tests. This model is useful to support wine tasting evaluations. Quality evaluation is part of wine certification process and can be used to improve wine making and classify wines to premium brands which can be useful for setting prices.

  The data set for this project is related to red and white vinho verde wine samples, from Portugal, created by P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. It is sourced from the UCI Machine Learning Repository and can be found and is available [here](https://archive.ics.uci.edu/ml/datasets/wine+quality). Each row in the data set represents label of wine (red or white) and its physicochemical properties which includes fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates.Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).

  To answer the predictive question posed above, we plan to build a predictive classification model. As the first step towards building the model we will split the data into train and test data set(split 80%, 20%).Then perform exploratory data analysis to determine and analyze the distribution of the features and correlation of features(physicochemical properties) with target classs(quality). 

  Thus far we have performed exploratory data analysis, and the report can be found [here](https://github.com/UBC-MDS/DSCI_522_Group19_Wine_Quality_Score_Predictor/blob/main/src/Wine_Score_EDA.ipynb).

## References

Paulo Cortez, University of Minho, Guimarães, Portugal, <http://www3.dsi.uminho.pt/pcortez> A. Cerdeira, F. Almeida, T. Matos and J. Reis, Viticulture Commission of the Vinho Verde Region(CVRVV), Porto, Portugal @2009 <https://archive.ics.uci.edu/ml/datasets/wine+quality>

P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.
