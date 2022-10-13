# Credit_Risk_Analysis
This project was to use machine learning to predict credit risk using an unbalanced dataset. 

## Software
 - python 3.9.12
     - pandas 1.3.5
     - numpy 1.21.5
     - scikit-learn 1.0
     - imbalanced-learn 0.9.0
 - jupyter notebook


## The data:
Lets look at the data before looking at the models used.    

### First 20 Rows
![](/Images/data_first_20_rows.png)    

### Dataframe Info
![](/Images/data_info.png)

68817 rows and 86 columns. There is some incorrect data types that need cleaned aswell.


## The Regression Models
6 supervised ML models were chosen:
- Scikit-learn's Logistic Regression was used with four different sampling techniques from imbalanced-learn:
    - RandomOverSampler 
    - SMOTE over sampler
    - ClusterCentroids under sampler
    - SMOTEEN
- 2 Imbalanced-learn ensemble classification models were also used with scikit-learn's train_test_split sampling:
    - BalancedRandomForestClassifier
    - EasyEnsembleClassifier 

## Results
The models differ in the sampling used for the training and use the same X values to test their performance. When you are comparing models that have the same output you can use the Adjusted Rand score of the model to help determine which is better, not just the rand score. If the models had different outputs then this metric would not be able to be compared. Below are the summary tables for each regression models performance.

### Scikit-learn's Logistic Regression
![](/Images/over_sampled_model.png)
![](/Images/smote_model.png)
![](/Images/under_sampled_model.png)
![](/Images/smoteen_model.png)

The SMOTE model is the best of these four, however the Adjusted Rand score is not high enough to consider any of them for production level code. I believe the Adjusted Rand Score is so low because the sensitivity detection for high risk is nearly non-existant. Finding a different model is highly recommended.

### Imbalanced-learn's Ensemble Classifiers
![](/Images/br_forest.png)
![](/Images/EE_model.png)

The Easy Ensemple model has the best Adjusted Rand Score of all 6 ML Regressions. It is the best option that was tested, but I still highly recommend using a different model. 

## Next Steps
Continuing with this project would be to explore the different sampling methods used on the scikit-learn model with imbalanced-learn's Ensemble Classifiers. The first test would be to combined the SMOTE sampling technique with the EasyEnsembleClassifier model, because they had the best performance in their own models.