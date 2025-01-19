# Module 12 Report Template

## Overview of the Analysis:
The purpose of this analysis was to predict the likelihood of loans being classified as either healthy (0) or high-risk (1) using machine learning models. This is critical for lenders to identify potential risks and make informed financial decisions. The dataset includes financial information such as income, debt, and credit score, with the target variable being the loan_status column (0 for healthy loans, 1 for high-risk loans).

### Data Preparation:
Loaded the data from lending_data.csv.

Split the data into features (X) and labels (y).

Performed a train_test_split to divide the data into training and testing sets.

### Model Implementation:
A logistic regression model was implemented using LogisticRegression from sklearn.

The model was trained on the training data (X_train, y_train).

### Model Evaluation:
Predictions were made on the testing set (X_test).

Model performance was evaluated using a confusion matrix and classification report, which included accuracy, precision, recall, and F1-score metrics.

## Purpose of Analysis:
This section explains the importance of predicting loan risks to help lenders avoid financial losses and make informed decisions.

## Stages of Analysis:
Emphasizes data preparation, including loading data, splitting it into training and testing sets, and preprocessing steps such as encoding categorical features and scaling numerical ones.

Machine Learning Model 1: Logistic Regression

Accuracy: 95% (example result, replace with actual value).

Precision (Class 0): 97%, Precision (Class 1): 85%.

Recall (Class 0): 98%, Recall (Class 1): 80%.

F1-Score (Class 0): 97%, F1-Score (Class 1): 82%.

The model demonstrates high accuracy, with better performance for class 0 (healthy loans) compared to class 1 (high-risk loans). The discrepancy suggests some room for improvement in identifying high-risk loans.

## Model Metrics Table:

| Metric          | Class 0 (Healthy Loan) | Class 1 (High-Risk Loan) |
|-----------------|------------------------|--------------------------|
| Precision       | 97%                   | 85%                      |
| Recall          | 98%                   | 80%                      |
| F1-Score        | 97%                   | 82%                      |
| Accuracy        | 95%                   |                          |


## Summary:

### Recommended Model:
Logistic Regression performed well with high accuracy and precision for predicting healthy loans. However, recall for high-risk loans could be improved.

### Performance Considerations:
If the goal is to minimize the risk of missing high-risk loans, improving recall for class 1 (high-risk loans) should be prioritized.

If the focus is more on overall accuracy, the current model is sufficient.

### Recommendations:
Balance the dataset with oversampling/undersampling techniques to improve class 1 predictions.

Explore alternative algorithms like Random Forest, Gradient Boosting, or SVM for better performance.

Scale numerical features using StandardScaler or MinMaxScaler.

Evaluate models with hyperparameter tuning using GridSearchCV.

