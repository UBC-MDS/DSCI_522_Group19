# Wine Quality Score Predictor

Contributors: Kingslin Lv, Manju Neervaram Abhinandana Kumar, Zack Tang, Pavel Levchenko

Wine Quality Score Predictor is our data analysis project for the 2021-22 UBC MDS DSCI 522 course.

## Introduction

The aim of this project is to predict the quality of wine on a scale of 0 to 10 given a set of physiochemical features rated by wine test reviewers as inputs. This model is useful to support wine tasting evaluations. The data set for this project is related to red and white vinho verde wine samples, from Portugal, created by P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. It is sourced from the UCI Machine Learning Repository and can be found [here](https://archive.ics.uci.edu/ml/datasets/wine+quality). Each row in the data set represents label of wine (red or white) and its physicochemical properties which includes fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, and sulphates.

We built a regression model using Ridge, One-Vs-Rest Logistic Regression, SVC, and Random Forest Regressor. Running through the cross-validation, we found the Random Forest Regressor delivers a much higher training score, but there was a clear case of overfitting issue. We then ran hyperparameter optimization in an attempt to improve the model. Unfortunately, the test score with the best hyperparameters was only around 0.53. By analyzing feature coefficients and we reduced to have 10 features. Some features have low coefficients as what was expected from our initial [EDA](https://github.com/UBC-MDS/DSCI_522_Group19_Wine_Quality_Score_Predictor/blob/main/src/Wine_Score_EDA.ipynb). In the coming weeks, we intend to refine our model further and come out a higher test score if possible.

## Report

The final report can be found [here](https://github.com/UBC-MDS/DSCI_522_Group19_Wine_Quality_Score_Predictor/blob/main/doc/Wine_Quality_Score_Predictor_report.md)

## Usage

To replicate the analysis, clone this GitHub repository, install the [dependencies](#dependencies) listed below, and run the following commands at the command line/terminal from the root directory of this project:

    # download wine data set to directory
    python src/download_data.py --url="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv" --out_file="data/raw/winequality-red.csv"
    python src/download_data.py --url="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv" --out_file="data/raw/winequality-white.csv"


    # pre-process data and split data to training set and test set
    python src/clean_split.py --input_red="data/raw/winequality-red.csv" --input_white="data/raw/winequality-white.csv" --out_dir="data/processed/"


    # create exploratory data analysis figure and write to file
    python src/Wine_Score_EDA.py --input_file="data/processed/train_df.csv"


    # fitting model and generating final results on the test data
    python src/model_fitting.py --X_train_path="data/processed/X_train.csv" --X_test_path="data/processed/X_test.csv" --y_train_path="data/processed/y_train.csv" --y_test_path="data/processed/y_test.csv"

    # render final report (RStudio terminal)
    Rscript -e "rmarkdown::render('reports/reports.Rmd', output_format = 'github_document')"

## Dependencies {#dependencies}

To run this project, please install

-   Python version 3.8.6 and the required dependencies from [here](https://github.com/UBC-MDS/DSCI_522_Group19_Wine_Quality_Score_Predictor/blob/main/env-wine-prediction.yaml) by running the following command at the command line/terminal

To set up the necessary packages for running the data analysis materials from wine score prediction, [download the environment file from the repo to your computer](https://github.com/UBC-MDS/DSCI_522_Group19_Wine_Quality_Score_Predictor/blob/main/env-wine-prediction.yaml) (hit "Raw" and then `Ctrl` + `s` to save it, or copy paste the content). Then create a Python virtual environment by using `conda` with the environment file you just downloaded:

``` bash
# create a conda environment using the `env-wine-prediction.yaml`
conda env create --file env-wine-prediction.yaml
conda activate wine
```

-   R version 4.0.2. and R packages:

    -   knitr==1.30

    -   feather==0.3.5

    -   kableExtra==1.3.1

    -   tidyverse==1.3.0

    -   docopt==0.6.2

## License

Datasets of this project are licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license. This allows for the sharing and adaptation of the datasets for any purpose, provided that the appropriate credit is given.
