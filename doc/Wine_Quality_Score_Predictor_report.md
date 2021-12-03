Predicting wine quality score from various characteristics
================
Group 19 - Kingslin Lv, Manju Neervaram Abhinandana Kumar, Zack Tang,
Paval Levchenko

-   [Summary](#summary)
-   [Introduction](#introduction)
-   [Methods](#methods)
    -   [Data](#data)
    -   [Analysis](#analysis)
-   [Results & Discussion](#results--discussion)
-   [Limitations & Future](#limitations--future)
-   [References](#references)

# Summary

In this project we aim to predict the wine quality scores ranging from 0
to 10 based on physicochemical properties of wines and sensory tests. To
answer this predictive question, we decided to build a regression model.

Through our exploratory data analysis we analyzed the distribution of
each feature and correlation between features and the target. Followed
by the cross-validation process based on feature input, we concluded
that the Random Forest Regressor delivers a much higher training score,
but there was a clear problem of overfitting. We further conducted
feature selection and hyperparameter optimization in an attempt to
reduce the score gap between train and test scores. We were able to drop
a number of features but maintain a relatively similar score through
this process. Unfortunately, the test score with our best
hyperparameters was only around 0.532, which is fairly acceptable. Next,
we can potentially improve our model prediction score by using a larger
dataset with more features or build a higher score model with its best
hyperparameters.


# Introduction

The wine industry shows a recent extensive growth and the industry
experts are using product quality certifications to promote their
products(Orth and Krška 2001). This is a time-consuming process and
requires the assessment given by human experts, which makes this process
very expensive. The wine market would be of interest if the human
quality of tasting can be related to wine’s chemical properties so that
quality assessment processes are more controlled. This project aims to
build a machine learning model for purpose of predicting the wine
quality score based on each of its specific chemical properties. This
task will likely require a lot of domain knowledge and according to a
paper published by Dr. P. Cortez, Dr. A. Cerdeira, Dr. F. Almeida,
Dr. T. Matos and Dr. J. Reis they were able to demonstrate the data
mining approach could have a promising result compared to alternative
neural network methods (Cortez et al. 2009).

Our model is useful to support wine tasting evaluations. Quality
evaluation is a part of wine certification process and can be used to
improve wine making or spot premium wines for a more proper price
according to customer taste preferences. Additionally, using human taste
as a sensory measurement for wine quality could be quite unreliable (De
Mets et al. 2017). We are also interested in exploring to what extent
the score depends on other sensory information such as color of wine.
Potentially, human brain could be processing taste and visual
information differently rather than taste only. Thus, we are not
expecting to obtain a really high test score to our machine learning
model.

# Methods

## Data

The dataset used in this project is retrieved from the University of
California Irvine (UCI) machine learning repository (Dua and Graff 2017)
and was collected by Paulo Cortez, University of Minho, Guimarães,
Portugal and A. Cerdeira, F. Almeida, T. Matos with help from J. Reis,
Viticulture Commission of the Vinho Verde Region(CVRVV), Porto, Portugal
in 2009. This dataset contains the results of various physiochemical
test, including scoring for properties like fixed acidity, volatile
acidity, citric acid, residual sugar, chlorides, free sulfur dioxide,
total sulfur dioxide, density, pH, sulphates, and alcohol (11 features),
which were preformed on white “Vinho Verde” wine samples from Northern
Portugal. The data used in our analysis can be found
[here](https://archive.ics.uci.edu/ml/datasets/wine+quality).
Additionally, we add one more feature by concatenating white and red
wine data, and so there is a binary feature; we think potentially
human’s perception of wine type may affect the independent scoring on
the wine quality, and so we added a binary feature to account for this
factor. Thus, there are 5197 instances and 12 features upon we combined
both red and white wine data.

One of drawback of our raw data is that there is no additional feature
or specific branding of each wine available in the dataset for privacy
purposes. Each row in the dataset represents a single wine which was
tested and scored based on human sensory data.

## Analysis

As the first step towards building the model to answer the predictive

similar scores with lesser features as displayed in the Table 1. This
process simplified our model and is cost-efficient for future data
collection.

<table class="table" style="width: auto !important; margin-left: auto; margin-right: auto;">
<caption>
Table 1. Cross-validation results for each model
</caption>
<thead>
<tr>
<th style="text-align:left;">
…1
</th>
<th style="text-align:left;">
Ridge
</th>
<th style="text-align:left;">
SVC
</th>
<th style="text-align:left;">
OneVsRest
</th>
<th style="text-align:left;">
Random Forest
</th>
<th style="text-align:left;">
Random Forest_rfe
</th>
</tr>
</thead>
<tbody>

</tr>
<tr>
<td style="text-align:left;">
test_neg_mean_squared_error
</td>
<td style="text-align:left;">
-0.542 (+/- 0.034)
</td>
<td style="text-align:left;">
-0.587 (+/- 0.016)
</td>
<td style="text-align:left;">
-0.648 (+/- 0.006)
</td>
<td style="text-align:left;">
-0.395 (+/- 0.027)
</td>
<td style="text-align:left;">
-0.395 (+/- 0.031)
</td>
</tr>
<tr>
<td style="text-align:left;">
train_neg_mean_squared_error
</td>
<td style="text-align:left;">
-0.536 (+/- 0.009)
</td>
<td style="text-align:left;">
-0.542 (+/- 0.007)
</td>
<td style="text-align:left;">
-0.642 (+/- 0.005)
</td>
<td style="text-align:left;">
-0.056 (+/- 0.002)
</td>
<td style="text-align:left;">
-0.056 (+/- 0.002)
</td>
</tr>
<tr>
<td style="text-align:left;">
test_neg_root_mean_squared_error
</td>
<td style="text-align:left;">
-0.736 (+/- 0.023)
</td>
<td style="text-align:left;">
-0.766 (+/- 0.010)
</td>
<td style="text-align:left;">
-0.805 (+/- 0.004)
</td>
<td style="text-align:left;">
-0.628 (+/- 0.021)
</td>
<td style="text-align:left;">
-0.628 (+/- 0.025)
</td>
</tr>
<tr>
<td style="text-align:left;">
train_neg_root_mean_squared_error
</td>
<td style="text-align:left;">
-0.732 (+/- 0.006)
</td>
<td style="text-align:left;">
-0.736 (+/- 0.005)
</td>
<td style="text-align:left;">
-0.801 (+/- 0.003)
</td>
<td style="text-align:left;">
-0.237 (+/- 0.003)
</td>
<td style="text-align:left;">
-0.238 (+/- 0.003)
</td>
</tr>
<tr>
<td style="text-align:left;">
test_neg_mean_absolute_error
</td>
<td style="text-align:left;">
-0.570 (+/- 0.015)
</td>
<td style="text-align:left;">
-0.480 (+/- 0.010)
</td>
<td style="text-align:left;">
-0.520 (+/- 0.003)
</td>
<td style="text-align:left;">
-0.450 (+/- 0.012)
</td>
<td style="text-align:left;">
-0.451 (+/- 0.015)
</td>
</tr>
<tr>
<td style="text-align:left;">
train_neg_mean_absolute_error
</td>
<td style="text-align:left;">
-0.567 (+/- 0.004)
</td>
<td style="text-align:left;">
-0.441 (+/- 0.004)
</td>
<td style="text-align:left;">
-0.514 (+/- 0.002)
</td>
<td style="text-align:left;">
-0.169 (+/- 0.002)
</td>
<td style="text-align:left;">
-0.169 (+/- 0.002)
</td>
</tr>
<tr>
<td style="text-align:left;">
test_r2
</td>
<td style="text-align:left;">
0.289 (+/- 0.029)
</td>
<td style="text-align:left;">
0.231 (+/- 0.020)
</td>
<td style="text-align:left;">
0.150 (+/- 0.006)
</td>
<td style="text-align:left;">
0.482 (+/- 0.027)
</td>
<td style="text-align:left;">
0.481 (+/- 0.032)
</td>
</tr>
<tr>
<td style="text-align:left;">
train_r2
</td>
<td style="text-align:left;">
0.298 (+/- 0.007)
</td>
<td style="text-align:left;">
0.290 (+/- 0.011)
</td>
<td style="text-align:left;">
0.158 (+/- 0.008)
</td>
<td style="text-align:left;">
0.926 (+/- 0.002)
</td>
<td style="text-align:left;">
0.926 (+/- 0.002)
</td>
</tr>
</tbody>
</table>

Finally, we conducted hyperparameter optimization as
`RandomForestRegressor` encountered severe overfitting issue. The best
hyperparameters we obtained from the algorithm are `max_depth` at 344,
`max_leaf_nodes` at 851, and `n_estimators`at 258 The best
cross-validation score is 0.483 using the best hyperparameter. The score
for test data set is 0.532 upon tuning hyper-parameters; however, as we
discovered above, the train score is 0.913 as displayed in the Table 2,
which indicates that we still have overfitting issue for the
`RandomForestRegressor` model.

<table class="table" style="width: auto !important; margin-left: auto; margin-right: auto;">
<caption>
Table 2. RandomForestRegressor model test results.
</caption>
<thead>
<tr>
<th style="text-align:left;">
best_model
</th>
<th style="text-align:right;">
RandomForestRegressor
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
max_depth
</td>
<td style="text-align:right;">
344.000
</td>
</tr>
<tr>
<td style="text-align:left;">
max_leaf_nodes
</td>
<td style="text-align:right;">
851.000
</td>
</tr>
<tr>
<td style="text-align:left;">
n_estimators
</td>
<td style="text-align:right;">
258.000
</td>
</tr>
<tr>
<td style="text-align:left;">
cv_best_score
</td>
<td style="text-align:right;">
0.483
</td>
</tr>
<tr>
<td style="text-align:left;">
train_score
</td>
<td style="text-align:right;">
0.913
</td>
</tr>
<tr>
<td style="text-align:left;">
test_score
</td>
<td style="text-align:right;">
0.532
</td>
</tr>
</tbody>
</table>

# Limitations & Future

The wine classification is a challenging task as it relies on sensory
analysis performed by human tasters. These evaluations are based on the
experience and knowledge of experts which are prone to be subjective
factors. One of main limitation here is that the dataset is imbalanced.
The majority of quality scores were 5 and 6. Another limitation is that
the dataset has only 12 features with one of binary feature that seems
not to add any values to our model. We could also potentially find a
larger dataset (i.e.with wine from different parts of the world) or with
more features since the one we are currently working with has a limited
number of features (i.e. lack of type of grape used in the wine) due for
the sake of privacy protection.

# References

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-rmarkdown" class="csl-entry">

Allaire, JJ, Yihui Xie, Jonathan McPherson, Javier Luraschi, Kevin
Ushey, Aron Atkins, Hadley Wickham, Joe Cheng, Winston Chang, and
Richard Iannone. 2020. *Rmarkdown: Dynamic Documents for r*.
<https://github.com/rstudio/rmarkdown>.

</div>

<div id="ref-pandasprofiling2019" class="csl-entry">

Brugman, Simon. 2019. “<span class="nocase">pandas-profiling:
Exploratory Data Analysis for Python</span>.”
<https://github.com/pandas-profiling/pandas-profiling>.

</div>

<div id="ref-CORTEZ2009547" class="csl-entry">

Cortez, Paulo, Antonio Cerdeira, Fernando Almeida, Telmo Matos, and Jose
Reis. 2009. “Modeling Wine Preferences by Data Mining from
Physicochemical Properties.” *Decision Support Systems* 47 (4): 547–53.
https://doi.org/<https://doi.org/10.1016/j.dss.2009.05.016>.

</div>

<div id="ref-docopt" class="csl-entry">

de Jonge, Edwin. 2020. *Docopt: Command-Line Interface Specification
Language*. <https://CRAN.R-project.org/package=docopt>.


De Mets, Guido, Peter Goos, Maarten Hertog, Christian Peeters, Jeroen
Lammertyn, and Bart M Nicolaı̈. 2017. “Sensory Quality of Wine: Quality
Assessment by Merging Ranks of an Expert-Consumer Panel.” *Australian
Journal of Grape and Wine Research* 23 (3): 318–28.


Dua, Dheeru, and Casey Graff. 2017. “UCI Machine Learning Repository.”
University of California, Irvine, School of Information; Computer
Sciences. <http://archive.ics.uci.edu/ml>.

</div>

<div id="ref-orth2001quality" class="csl-entry">

Orth, Ulrich R, and Pavel Krška. 2001. “Quality Signals in Wine
Marketing: The Role of Exhibition Awards.” *The International Food and
Agribusiness Management Review* 4 (4): 385–97.

</div>

<div id="ref-scikit-learn" class="csl-entry">

Pedregosa, F., G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O.
Grisel, M. Blondel, et al. 2011. “Scikit-Learn: Machine Learning in
Python.” *Journal of Machine Learning Research* 12: 2825–30.

</div>

<div id="ref-R" class="csl-entry">

R Core Team. 2019. *R: A Language and Environment for Statistical
Computing*. Vienna, Austria: R Foundation for Statistical Computing.
<https://www.R-project.org/>.

</div>

<div id="ref-reback2020pandas" class="csl-entry">

team, The pandas development. 2020. *Pandas-Dev/Pandas: Pandas* (version
latest). Zenodo. <https://doi.org/10.5281/zenodo.3509134>.

</div>

<div id="ref-Python" class="csl-entry">

Van Rossum, Guido, and Fred L. Drake. 2009. *Python 3 Reference Manual*.
Scotts Valley, CA: CreateSpace.

</div>

<div id="ref-knitr" class="csl-entry">

Xie, Yihui. 2020. *Knitr: A General-Purpose Package for Dynamic Report
Generation in r*. <https://yihui.org/knitr/>.

</div>

</div>
