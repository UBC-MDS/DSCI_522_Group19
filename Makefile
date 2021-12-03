# DSCI_522_Group19_Wine_Quality_Score_Predictor

# author: DSCI 522 Group 19
# date: 2021-12-01
# 
# usage: make all

all: results/quality_dist.png results/repeat_plots.png results/cor_plot.png results/cv_scores_for_alternative_methods.csv results/final_results.csv doc/Wine_Quality_Score_Predictor_report.md doc/Wine_Quality_Score_Predictor_report.html
  
# download data and save as csv
data/raw/winequality-red.csv : src/download_data.py
	python src/download_data.py --url="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv" --out_file="data/raw/winequality-red.csv"
data/raw/winequality-white.csv : src/download_data.py
	python src/download_data.py --url="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv" --out_file="data/raw/winequality-white.csv"

# pre-process and split data to training set and test set
data/processed/train_df.csv data/processed/test_df.csv data/processed/X_train.csv data/processed/X_test.csv data/processed/y_train.csv data/processed/y_test.csv : src/clean_split.py data/raw/winequality-red.csv data/raw/winequality-white.csv 
	python src/clean_split.py --input_red="data/raw/winequality-red.csv" --input_white="data/raw/winequality-white.csv" --out_dir="data/processed/"

# create exploratory data analysis figures
results/quality_dist.png results/repeat_plots.png results/cor_plot.png : src/Wine_Score_EDA.py data/processed/train_df.csv 
	python src/Wine_Score_EDA.py --input_file="data/processed/train_df.csv" --out_dir="results"
	
# fit, tune and test model
results/cv_scores_for_alternative_methods.csv results/final_results.csv : src/model_fitting.py
	python src/model_fitting.py --X_train_path="data/processed/X_train.csv" --X_test_path="data/processed/X_test.csv" --y_train_path="data/processed/y_train.csv" --y_test_path="data/processed/y_test.csv"

# render final report
doc/Wine_Quality_Score_Predictor_report.md doc/Wine_Quality_Score_Predictor_report.html: doc/references_wine_score_predictor.bib doc/Wine_Quality_Score_Predictor_report.Rmd
	Rscript -e "rmarkdown::render('doc/Wine_Quality_Score_Predictor_report.Rmd', output_format = 'github_document')"
	
clean: 
	rm -rf data/raw/winequality-red.csv
	rm -rf data/raw/winequality-white.csv
	rm -rf data/processed/X_test.csv
	rm -rf data/processed/X_train.csv
	rm -rf data/processed/Xtrain_transformed.csv
	rm -rf data/processed/test_df.csv
	rm -rf data/processed/train_df.csv
	rm -rf data/processed/y_test.csv
	rm -rf data/processed/y_train.csv
	rm -rf results/quality_dist.png
	rm -rf results/repeat_plots.png
	rm -rf results/cor_plot.png
	rm -rf results/cv_scores_for_alternative_methods.csv
	rm -rf results/final_results.csv	
	rm -rf doc/Wine_Quality_Score_Predictor_report.md 
	rm -rf doc/Wine_Quality_Score_Predictor_report.html	