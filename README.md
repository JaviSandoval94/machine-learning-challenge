# Machine Learning Challenge - Exoplanet Exploration

![exploanets.jpg](Images/exoplanets.jpg)

## Background
Over a period of nine years in deep space, the NASA Kepler space telescope has been out on a planet-hunting mission to discover hidden planets outside of our solar system.

To help process this data, we will build machine learning models capable of classifying candidate exoplanets from a raw dataset.
## Data set
All data comes from the Kepler Exoplanet Search Results dataset, available on Kaggle at https://www.kaggle.com/nasa/kepler-exoplanet-search-results. Further documentation on the specific content of each of the fields contained in the data set can be found here: https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html

The data columns were classified into two different groups:
- Non-categorical: These are features allowing continuous values.
- Categorical: Every feature containing less than 10 distinct values is labeled as a categorical feature.

Further discussion of such features, as well as a characterization of all the relevant features found in the dataset, can be found in the `data_exploration.ipynb` file.

## Models generated
All the models were generated using the latest Sklearn and Joblib packages avabilable as of 1st August 2020. The classification models were made from the provided dataset using the `koi_disposition` feature as predicted values. In all cases, the dataset was filtered to contain only the `CONFIRMED` and `FALSE POSITIVE`outcomes, leaving out of the model all the classification pending records (labeled as `CANDIDATE` in the original dataset).

Four classification models were considered:
- Model 1: Classification Trees and Random Forest Classifiers
- Model 2: Support Vector Machines
- Model 3: K-Nearest Neighbors
- Model 4: Logistic Regression Classifier

In all cases, the available data was preprocessed and scaled into train-test samples and tuned using the GridSearchCV method (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html). For every model, the GridSearchCV method was used to obtain the best estimator along with its score. Finally, the model was refitted using the best estimator to calculate final testing and training scores. These were the results obtained for each model:

| Notebook | Model                    | Best fit parameters                                                                                                             | GridSearchCV best score | Training score | Testing score |
|----------|--------------------------|-----------------------------------------------------------------------------------------------------------------------------|-------------------------|----------------|---------------|
| Model 1  | Random Forest Classifier | {'bootstrap': True, 'max_depth': 90, 'max_features': 3, 'min_samples_leaf': 4, 'min_samples_split': 8, 'n_estimators': 300} | 0.9738                  | 1.0            | 0.9773        |
| Model 2  | Support Vector Machine   | {'C': 1, 'gamma': 0.0001}                                                                                                   | 0.9919                  | 0.9919         | 0.9856        |
| Model 3  | K-Nearest Neighbors      | {'metric': 'manhattan', 'n_neighbors': 5, 'weights': 'distance'}                                                            | 0.9922                  | 1.0            | 0.9864        |
| Model 4  | Logistic Regression      | {'C': 0.01, 'penalty': 'l2'}                                                                                                | 0.9919                  | 0.9919         | 0.9856        |


## Final model discussion
Model 3 was selected as the final model and saved in the `final_model.sav` file. Though this model did not get the highest score out of the best estimators, it did obtain the highest testing score. This parameter was preferred over GridSearchCV and training scores to prevent model overfitting. This model thus seems to be the best one to predict outcomes from new values.

The final model makes the following assumptions:
- The `koi_tce_plnt_num` was excluded of the analysis. This model should thus not be used if the TCE Planet Number is relevant in any context.
- This model does not consider records labeled as `CANDIDATE` in the KOI classification. The KOI dataset is regularly updated and these results are very likely to change as the records are updated. Furthermore, fit results might be sensibly different after the candidate features are included.

These models should not be used outside of a demonstration context. Further tuning of the models obtained should consider NASA updates in the KOI dataset and be regularly revisited to ensure that the latest dataset and packages versions are used.
