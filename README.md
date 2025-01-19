# credit-risk-classification
## Loan Risk Analysis Code
### Purpose
The objective of this project is to predict loan risk levels using a logistic regression model. By classifying loans as either healthy (0) or high-risk (1), this analysis helps lenders identify potential risks and take informed financial decisions.

## Workflow
## Data Loading:

The dataset is imported from lending_data.csv, which contains financial indicators such as income, debt-to-income ratio, and credit score.
The target variable, loan_status, indicates whether a loan is healthy (0) or high-risk (1).

## Data Preparation:
The dataset is split into features (X) and labels (y).
A train-test split is performed to divide the data into training and testing sets, ensuring reproducibility by setting a random_state.

### Feature Scaling:
Feature values are scaled to improve model performance and ensure faster convergence during training.

### Model Training:
A logistic regression model is trained on the training dataset. The max_iter parameter is adjusted to ensure the model converges successfully.

### Prediction:
Predictions are made on the test dataset to assess how well the model generalizes to unseen data.

### Evaluation:
Model performance is evaluated using a confusion matrix and a classification report. Metrics such as accuracy, precision, recall, and F1-score are calculated for both healthy and high-risk loan classes.

## Results
Accuracy: The model achieved an accuracy of approximately 95%, indicating strong overall performance.
### Class-Specific Metrics:

#### Healthy Loans (Class 0): 
The model showed excellent precision (97%) and recall (98%), meaning most healthy loans were correctly identified.
#### High-Risk Loans (Class 1): 
While the precision (85%) was satisfactory, recall (80%) highlighted areas for improvement in identifying all high-risk loans.

### Key Findings:
The model tends to perform better for the majority class (healthy loans), which suggests potential bias due to class imbalance in the dataset.

## Key Insights
### Strengths:
The logistic regression model is simple, interpretable, and performs well with minimal preprocessing.
High accuracy and precision make it reliable for predicting healthy loans.
### Weaknesses:
Lower recall for high-risk loans means the model misses some high-risk predictions, which could lead to financial losses for lenders.

### Next Steps:
##### Address Class Imbalance:
Use oversampling techniques like SMOTE or undersampling to balance the dataset and improve predictions for high-risk loans.
##### Experiment with Advanced Models:
Evaluate other machine learning algorithms such as Random Forest, Gradient Boosting, or Support Vector Machines (SVM) for potentially better performance.
##### Optimize Hyperparameters:
Use GridSearchCV or RandomizedSearchCV to fine-tune the logistic regression model and explore different solvers and regularization methods.
##### Incorporate Feature Engineering:
Add new features or transform existing ones to improve the model’s ability to differentiate between healthy and high-risk loans.
## Conclusion
This analysis highlights the strengths of logistic regression in predicting loan risks but also points to areas of improvement, especially in identifying high-risk loans. Future work will focus on addressing class imbalance, enhancing the dataset, and exploring alternative machine learning models for better performance.
