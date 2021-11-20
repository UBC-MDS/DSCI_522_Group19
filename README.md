# Wine Quality Score Predictor

-   Contributors: Kingslin Lv, Manju Neervaram Abhinandana Kumar, Zack Tang, Paval Levchenko

Wine Quality Score Predictor is our data analysis project for the 2021-22 UBC MDS DSCI 522 course.

## Introduction

In this project, we aim to predict the wine taste preferences as quality scores ranging from 0 to 10 based on physicochemical properties of the wines and sensory tests. This model is useful to support wine tasting evaluations. Quality evaluation is part of wine certification process and can be used to improve wine making and classify wines to premium brands which can be useful for setting prices and for marketing purposes based on consumer tastes. It should be noted that using taste as a sensory measurement for wine quality could be quite unreliable. So we are also interested in exploring how much output data could depend on other sensory information such as color of wine. Potentially, human brain could be processing taste and visual information differently rather than taste only. Thus, this project aims to predict quality score based on features provided. 

The data set for this project is related to red and white vinho verde wine samples, from Portugal, created by P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. It is sourced from the UCI Machine Learning Repository and can be found and is available [here](https://archive.ics.uci.edu/ml/datasets/wine+quality). Each row in the data set represents label of wine (red or white) and its physicochemical properties which includes fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, and sulphates. Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).

To answer the predictive question posed above, we plan to build a predictive classification model. As the first step towards building the model we will split the data into train and test data set(split 80% and 20%). Then perform exploratory data analysis to determine and analyze the distribution of each feature and correlation between features and the target (wine scores ranging from 0 to 10).

Thus far we have performed exploratory data analysis, and the report can be found [here](https://github.com/UBC-MDS/DSCI_522_Group19_Wine_Quality_Score_Predictor/blob/main/src/Wine_Score_EDA.ipynb).
  
## Anlaysis Plan

For this project, we will use supervised learning techniques to approach our research problem. After exploring the data set through proper EDA, we decide to use `StandardScaler` to transform all numeric features since the variation of those features are quite high. For the one binary feature we made from the two data sets, we are going to use `OneHotEncoder` with `drop="if_binary"` argument to create one column for the "type" feature.
  
Following the `ColumnTransformer` process, we will dive deep into the class imbalance issue we figured out in the EDA process and add class_weight argument in the model. We will attempt to utilize three different models to fit the training data set. The tentative plan is to use decision tree, SVC, and KNN methodology. By applying cross-validation techniques and comparing validation and train scores, we are going to select the most well-performed model and conduct hyper-parameter optimization accordingly. 
  
Also, we will use different matrix to evaluate our model. For example, we consider to use confusion matrix to mainly assess the precision score, given we want to prevent customers from purchasing a poor wine while our model incorrectly predicts a higher wine score. 
  
After tuning the model, we will use test data set to do the final check on the accuracy. If the result is not satisfied, we will make more adjustments based on the new issue. 
  
## Usage

To set up the necessary packages for running the data analysis materials from wine score prediction,
[download the environment file from the repo to your computer](https://github.com/UBC-MDS/DSCI_522_Group19_Wine_Quality_Score_Predictor/blob/main/env-wine-prediction.yaml)
(hit "Raw" and then `Ctrl` + `s` to save it, or copy paste the content).
Then create a Python virtual environment by using `conda` with the environment file you just downloaded:

```
conda env create --file env-wine-prediction.yaml
```
  
## License

Datasets of this project are licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license. This allows for the sharing and adaptation of the datasets for any purpose, provided that the appropriate credit is given.

## References

Paulo Cortez, University of Minho, Guimarães, Portugal, <http://www3.dsi.uminho.pt/pcortez> A. Cerdeira, F. Almeida, T. Matos and J. Reis, Viticulture Commission of the Vinho Verde Region(CVRVV), Porto, Portugal @2009 <https://archive.ics.uci.edu/ml/datasets/wine+quality>

P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009. https://www.sciencedirect.com/science/article/abs/pii/S0167923609001377?via%3Dihub
