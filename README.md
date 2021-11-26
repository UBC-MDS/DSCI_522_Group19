# Wine Quality Score Predictor

Contributors: Kingslin Lv, Manju Neervaram Abhinandana Kumar, Zack Tang, Paval Levchenko

Wine Quality Score Predictor is our data analysis project for the 2021-22 UBC MDS DSCI 522 course.

## Introduction

In this project, the aim of this project is to predict the quality of wine on a scale of 0 to 10 given a set of physiochemical features rated by wine test reviewers as inputs. This model is useful to support wine tasting evaluations. Quality evaluation is part of wine certification process and can be used to improve wine making and classify wines to premium brands which can be useful for setting prices and for marketing purposes based on consumer tastes.

The data set for this project is related to red and white vinho verde wine samples, from Portugal, created by P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. It is sourced from the UCI Machine Learning Repository and can be found [here](https://archive.ics.uci.edu/ml/datasets/wine+quality). Each row in the data set represents label of wine (red or white) and its physicochemical properties which includes fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, and sulphates.

We built a regression model using Ridge, Logistic Regression, SVC, and Random Forest. Running through the cross-validation, we found the Random Forest delivers a much higher training score,but there was a clear case of overfitting issue. We then ran hyperparameter optimization in an attempt to improve the model. Unfortunately, the test score with the best hyperparameters was only around xxxxx. By analyzing feature coefficients and we can obtain XXXX feature had the highest coefficient score, which was expected from our initial [EDA] (https://github.com/UBC-MDS/DSCI_522_Group19_Wine_Quality_Score_Predictor/blob/main/src/Wine_Score_EDA.ipynb) In the coming weeks, we intend to refine our model further and come out a higher test score if possible. **some contents here are placeholder**

## Report

The final report can be here found [here] (https://github.com/UBC-MDS/DSCI_522_Group19_Wine_Quality_Score_Predictor/blob/main/doc/Wine_Quality_Score_Preidctor_report.Rmd)

  
## Usage

To replicate the analysis, clone this GitHub repository, install the
[dependencies](#dependencies) listed below, and run the following
commands at the command line/terminal from the root directory of this
project: 

```
# download wine data set to directory
python src/download_data.py --url="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv" --out_file="../data/raw/winequality-red.csv"
python src/download_data.py --url="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv" --out_file="data/raw/winequality-white.csv"


# pre-process data and split data to training set and test set
python src/preprocessing.py --input_red="../data/raw/winequality-red.csv" --input_white="../data/raw/winequality-white.csv" --out_file="../data/processed/preprocessed_Xtrain.csv"


# create exploratory data analysis figure and write to file
python Wine_Score_EDA.py --input_file="..data/processed/train_df.csv"


# fitting model
python src/fit_wine_quality_predict_model.py --in_file_1="data/processed/processed_train.csv" --out_dir="results/"

# test model
python src/wine_quality_test_results.py --in_file_1="data/processed/processed_train.csv" --in_file_2="data/processed/processed_test.csv" --out_dir="results/"


# render final report (RStudio terminal)
Rscript -e "rmarkdown::render('reports/reports.Rmd', output_format = 'github_document')"

```

## Dependencies

To run this project, please install 

* Python version 3.8.6 and the required dependencies from [here](https://github.com/UBC-MDS/DSCI_522_Group19_Wine_Quality_Score_Predictor/blob/main/env-wine-prediction.yaml) by running the following command at the command line/terminal

To set up the necessary packages for running the data analysis materials from wine score prediction,
[download the environment file from the repo to your computer](https://github.com/UBC-MDS/DSCI_522_Group19_Wine_Quality_Score_Predictor/blob/main/env-wine-prediction.yaml)
(hit "Raw" and then `Ctrl` + `s` to save it, or copy paste the content).
Then create a Python virtual environment by using `conda` with the environment file you just downloaded:

```bash
# create a conda environment using the `wine.yaml`
conda env create --file wine.yaml
conda activate wine
```

* R version 4.0.2. and R packages:
 - knitr==1.30
 - kableExtra==1.3.1
 - tidyverse==1.3.0
 - docopt==0.6.2
  
## License

Datasets of this project are licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license. This allows for the sharing and adaptation of the datasets for any purpose, provided that the appropriate credit is given.

## References

Paulo Cortez, University of Minho, Guimarães, Portugal, <http://www3.dsi.uminho.pt/pcortez> A. Cerdeira, F. Almeida, T. Matos and J. Reis, Viticulture Commission of the Vinho Verde Region(CVRVV), Porto, Portugal @2009 <https://archive.ics.uci.edu/ml/datasets/wine+quality>

P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009. https://www.sciencedirect.com/science/article/abs/pii/S0167923609001377?via%3Dihub
