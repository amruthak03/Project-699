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
2. Implemented 5 different attribute selection methods to identify optimal features. The 5 attribute selection methods used are:
   a. Principal Component Analysis (PCA)
   b. Linear Discriminant Analysis (LDA)
   c. Random Forest classifier
   d. XGBoost classifier
   e. Lasso Regression
3. Developed 5 different classification models, which are J48 Decision Tree, Naive Bayes classifier, Logistic Regression, Support Vector MAchine (SVM), and Neural Network, for each chosen set of attributes. 10-fold cross-validation was used while training the classification models.
4. Compared each of the 25 models built and trained based on their accuracies. Since we have a balanced dataset, accuracy was a good measure for the performance comparison of each model. The best combination of feature selection method and classification model was selected. The best performing classification model was then implemented on the original dataset without feature selection and the performances of the two models were compared. 

### Result
The best-performing model achieved an accuracy of 79% and an AUC of 0.8484 _(as shown in the graph below)_

<img width="416" alt="Screenshot 2023-08-06 at 3 22 44 PM" src="https://github.com/amruthak03/Project-699/assets/110037114/c3dcfa98-1a3f-451b-a66d-89efdc1ab078" class="center">
