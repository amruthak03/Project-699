# Binary Classification Techniques to Identify High or Low Graduation Rates in US Colleges
## Project Overview
This project focuses on employing various binary classification techniques to predict whether a US college has a high or low graduation rate. The project was completed as part of the Data Mining course at Boston University.

## Data Mining Tools Used
1. R Programming
2. caret
3. rpart
4. randomForest
5. xgboost
6. pROC
7. glmnet
8. Rsample

## Methodology
1. Data Preprocessing: Used KNN impute method to treat missing values and used undersampling method to balance the dataset.
2. Streamlined data analysis workflow, reducing feature dimensionality by 35% through effective feature selection methods leading to a 5% improvement in model accuracy as compared to the initial dataset.
3. Developed and evaluated 25 machine learning models using 5 classification algorithms and 5 feature selection methods to achieve an average accuracy of 80%, with the Random Forest model outperforming other models across multiple attribute selection methods.

The 5 attribute selection methods used are:
   a. Principal Component Analysis (PCA)
   b. Linear Discriminant Analysis (LDA)
   c. Random Forest classifier
   d. XGBoost classifier
   e. Lasso Regression

The 5 different classification models used are:
   a. J48 Decision Tree
   b. Naive Bayes classifier
   c. Logistic Regression
   d. Support Vector Machine (SVM)
   e. Neural Network
 
Since the dataset was balanced, accuracy was a good measure for the performance comparison of each model. The best combination of feature selection method and classification model was selected. The best-performing classification model was then implemented on the original dataset without feature selection and the performances of the two models were compared. 

### Result
The best-performing model achieved an accuracy of 80% and an AUC of 0.8484 _(as shown in the graph below)_

<center>
<img alt="Screenshot 2023-08-06 at 3 22 44 PM" src="https://github.com/amruthak03/Project-699/assets/110037114/c3dcfa98-1a3f-451b-a66d-89efdc1ab078" width="327">
</center>
