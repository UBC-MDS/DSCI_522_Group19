# Wine Quality Score Predictor

-   Contributors: Kingslin Lv, Manju Neervaram Abhinandana Kumar, Zack Tang, Paval Levchenko

Wine Quality Score Predictor is our Data analysis project for DSCI 522.

## Introduction

  In this project, we aim to predict the wine taste prefrences as quality scores ranging from 0 to 10 based on physicochemical properties of the wines and sensory tests. This model is useful to support wine tasting evaluations. Quality evaluation is part of wine certification process and can be used to improve wine making and classify wines to premium brands which can be useful for setting prices and for marketing purposes based on consumer tastes. It should be noted that using taste as a sensory measurement for wine quality could be quite unreliable. So we are also interested in exploring how much output data could depend on other sensory information such as color of wine. Potentially, human brain could be processing taste and visual information differently rather than taste only. 

  The data set for this project is related to red and white vinho verde wine samples, from Portugal, created by P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. It is sourced from the UCI Machine Learning Repository and can be found and is available [here](https://archive.ics.uci.edu/ml/datasets/wine+quality). Each row in the data set represents label of wine (red or white) and its physicochemical properties which includes fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates.Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).

  To answer the predictive question posed above, we plan to build a predictive classification model. As the first step towards building the model we will split the data into train and test data set(split 80%, 20%).Then perform exploratory data analysis to determine and analyze the distribution of the features and correlation of features(physicochemical properties) with target classs(quality). 

  Thus far we have performed exploratory data analysis, and the report can be found [here](https://github.com/UBC-MDS/DSCI_522_Group19_Wine_Quality_Score_Predictor/blob/main/src/Wine_Score_EDA.ipynb).

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

P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.
