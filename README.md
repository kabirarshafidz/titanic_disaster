
# Titanic Survival Prediction - Kaggle Competition

## Overview

This project focuses on predicting the survival of passengers aboard the Titanic using data from the [Kaggle Titanic competition](https://www.kaggle.com/c/titanic). By analyzing and processing the dataset, we build a classification model to predict which passengers survived the disaster.

## Objective

The main goal of this project is to create a predictive model that accurately classifies passengers as survivors or non-survivors. Using various data preprocessing, feature engineering, and machine learning techniques, we aim to maximize prediction accuracy on the test dataset.

## Dataset

The Titanic dataset includes the following:

- **Training Data**: Contains labeled data (Survived) with passenger details.
- **Test Data**: Contains unlabeled data used for predictions.
- **Target Variable**: `Survived` (1 if the passenger survived, 0 otherwise).

Key features in the dataset include:

- **Numerical Features**: Age, Fare, etc.
- **Categorical Features**: Sex, Pclass, Embarked, etc.

## Approach

1. **Data Preprocessing**:
   - Handle missing values in numerical and categorical features.
   - Impute data as needed and normalize or scale numerical features.
   - Encode categorical variables to prepare for model input.

2. **Feature Engineering**:
   - Create new features to enhance model performance.
   - Select important features based on statistical analysis and correlation with survival.

3. **Modeling and Ensemble Method**:
   - Train multiple classification models, including Logistic Regression, Random Forest, and Support Vector Machines.
   - Use ensemble techniques, such as Voting Classifier, to combine the strengths of different models for improved accuracy and generalization.

4. **Hyperparameter Tuning with Grid Search CV**:
   - Apply Grid Search Cross-Validation (Grid Search CV) to find optimal hyperparameters for the models.
   - Test combinations of parameters (e.g., max_depth, n_estimators, min_samples_split) to maximize model performance.
   - Choose the best hyperparameters based on cross-validation results, which improve model robustness and accuracy on the validation set.

## Evaluation Metric

The competition uses **Accuracy** as the evaluation metric, calculated as:

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

## Results

The final model achieved an accuracy of **84%** on the validation set and **77.751%** on the Kaggle leaderboard. Techniques such as feature engineering, model ensembling, and hyperparameter tuning helped enhance the model's performance.

## Conclusion

This project demonstrates the application of classification techniques to predict passenger survival. Future work could involve exploring additional features, improving imputation methods, and testing more advanced ensemble models for better accuracy.

## Acknowledgments

- Kaggle for hosting the competition and providing the dataset.
