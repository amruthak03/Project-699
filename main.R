library(caret)
#library(RANN)#required for knnImpute using preProcess
library(rsample)
library(rpart)
#library(MASS)
library(randomForest)
library(xgboost)
library(pROC)
#library(ROCR)
library(glmnet)

data = read.csv('colleges_usnews.csv')

head(data)

data[data == '?'] <- NA
nrow(data)

sapply(data, function(x) sum(is.na(x)))

set.seed(0)
data_new = dplyr::select(data, Average_Math_SAT_score : Instructional_expenditure_per_student)

data_new <- lapply(data_new, as.numeric)

data_new <- data.frame(data_new)

preProcessValue <- preProcess(data_new,
                              method = c('knnImpute'),
                              k = 10,
                              knnSummary = mean)
impute_data <- predict(preProcessValue, data_new, na.action = na.pass)
impute_data

procNames <- data.frame(col = names(preProcessValue$mean), 
                        mean = preProcessValue$mean,
                        sd = preProcessValue$std)
for (i in procNames$col){
  impute_data[i] <- impute_data[i]*preProcessValue$std[i] + preProcessValue$mean[i]
}
impute_data

colnames(data)

combined_dataset <- cbind(impute_data, data[, c(2,3,4,5,36)])
#We have not included the 'id' column since it is just giving us the index
combined_dataset

combined_dataset$binaryClass <- as.factor(combined_dataset$binaryClass)

cols_to_encode <- c('College_name', 'State')

label_encode <- function(df, cols_to_encode){
  for (col in cols_to_encode){
    df[[col]] <- as.numeric(factor(df[[col]]))
  }
  return(df)
}

combined_dataset <- label_encode(combined_dataset, cols_to_encode)

combined_dataset

table(combined_dataset$binaryClass)

sapply(combined_dataset, function(x) sum(is.na(x)))

#Creating a balanced dataset
min <- combined_dataset[combined_dataset$binaryClass == 'P',]
maj <- combined_dataset[combined_dataset$binaryClass == 'N',]

set.seed(0)
balanced <- rbind(min, maj[sample(1:nrow(maj), nrow(min)),])

# write.csv(balanced, "preProcessed.csv", row.names = FALSE)

#Splitting the dataset into train and test sets
set.seed(0)
split <- initial_split(balanced, prop = 0.66, strata = binaryClass)
train <- training(split)
test <- testing(split)

#Applying PCA on the training dataset
pc <- prcomp(train[, -35], center = TRUE, scale = TRUE) # exclude class attribute
summary(pc)

# Let’s use only first 14 which explain 90% of the variation
# first map (project) original attributes to new attributes (using the projection matrix)
# created by PCA
tr <- predict(pc, train)
tr <- data.frame(tr, train[35])
ts <- predict(pc, test)
ts <- data.frame(ts, test[35])

perf_measure <- function(cm){
  #class 'N'
  tp <- cm[1,1]
  tn <- cm[2,2]
  fn <- cm[2,1]
  fp <- cm[1,2]
  
  tpr <- tp / (tp+fn)
  fpr <- fp / (fp+tn)
  
  precision <- tp / (tp+fp)
  recall <- tpr
  
  f1 <- 2 * precision * recall / (precision + recall)
  
  mcc <- (tp * tn - fp * fn) / sqrt((tp + fp) * (tp+fn) * (tn+fp) * (tn+fn))
  
  df <- data.frame(Class = 'N',
                   TPR = tpr,
                   FPR = fpr,
                   Precision = precision,
                   Recall = recall,
                   F1_score = f1,
                   MCC = mcc)
  
  #class 'P'
  tn <- cm[1,1]
  tp <- cm[2,2]
  fp <- cm[2,1]
  fn <- cm[1,2]
  
  tpr <- tp / (tp+fn)
  fpr <- fp / (fp+tn)
  
  precision <- tp / (tp+fp)
  recall <- tpr
  
  f1 <- 2 * precision * recall / (precision + recall)
  
  mcc <- (tp * tn - fp * fn) / sqrt((tp + fp) * (tp+fn) * (tn+fp) * (tn+fn))
  
  df1 <- data.frame(Class = 'P',
                   TPR = tpr,
                   FPR = fpr,
                   Precision = precision,
                   Recall = recall,
                   F1_score = f1,
                   MCC = mcc)
  data <- rbind(df, df1)
  
  return(data)
}


j48_classifier <- function(train, test, class_data){
  set.seed(0)
  train_control <- trainControl(method = "cv", number = 10)
  model <- train(binaryClass ~ ., data = train, method = "J48", trControl = train_control)
  predicted <- predict(model, newdata = test)
  pred <- predict(model, newdata = test, type = 'prob')
  roc_curve <- roc(class_data, pred[,2])
  auc <- auc(roc_curve)
  plot(roc_curve)
  df <- perf_measure(table(predicted, class_data))
  cm <- confusionMatrix(predicted, class_data)
  return(list(cm, auc, df))
}

j48_classifier(tr[c(1:14, 35)], ts[c(1:14)], ts[,35])

nb_classifier <- function(train, test, class_data){
  set.seed(0)
  train_control <- trainControl(method = "cv", number = 10)
  model <- train(binaryClass ~ ., data = train, method = "nb", trControl = train_control)
  predicted <- predict(model, newdata = test)
  pred <- predict(model, newdata = test, type = 'prob')
  roc_curve <- roc(class_data, pred[,2])
  auc <- auc(roc_curve)
  plot(roc_curve)
  df <- perf_measure(table(predicted, class_data))
  cm <- confusionMatrix(predicted, class_data)
  return(list(cm, auc, df))
}

nb_classifier(tr[c(1:14, 35)], ts[c(1:14)], ts[,35])

logistic_classifier <- function(train, test, class_data){
  set.seed(0)
  train_control <- trainControl(method = "cv", number = 10)
  model <- train(binaryClass ~ ., data = train, method = "glm", family = 'binomial', trControl = train_control)
  predicted <- predict(model, newdata = test)
  pred <- predict(model, newdata = test, type = 'prob')
  roc_curve <- roc(class_data, pred[,2])
  auc <- auc(roc_curve)
  plot(roc_curve)
  df <- perf_measure(table(predicted, class_data))
  cm <- confusionMatrix(predicted, class_data)
  return(list(cm, auc, df))
}

logistic_classifier(tr[c(1:14, 35)], ts[c(1:14)], ts[,35])

svm_classifier <- function(train, test, class_data){
  set.seed(0)
  train_control <- trainControl(method = "cv", number = 10, 
                                classProbs = TRUE, 
                                summaryFunction = twoClassSummary)
  model <- train(binaryClass ~ ., 
                 data = train, 
                 method = "svmRadial",
                 preProc = c('center', 'scale'),
                 trControl = train_control,
                 metric = 'ROC')
  predicted <- predict(model, newdata = test)
  pred <- predict(model, newdata = test, type = 'prob')
  roc_curve <- roc(class_data, pred[,2])
  auc <- auc(roc_curve)
  plot(roc_curve)
  df <- perf_measure(table(predicted, class_data))
  cm <- confusionMatrix(predicted, class_data)
  return(list(cm, auc, df))
}

svm_classifier(tr[c(1:14, 35)], ts[c(1:14)], ts[,35])

nnet_classifier <- function(train, test, class_data){
  set.seed(0)
  train_control <- trainControl(method = "cv", number = 10,
                                summaryFunction = twoClassSummary,
                                classProbs = TRUE,
                                savePredictions = TRUE)
  model <- train(binaryClass ~ ., 
                 data = train, 
                 method = "nnet",
                 metric = 'ROC',
                 preProc = c('center', 'scale'),
                 trControl = train_control,
                 trace = FALSE,
                 maxit = 100)
  predicted <- predict(model, newdata = test)
  pred <- predict(model, newdata = test, type = 'prob')
  roc_curve <- roc(class_data, pred[,2])
  auc <- auc(roc_curve)
  plot(roc_curve)
  df <- perf_measure(table(predicted, class_data))
  cm <- confusionMatrix(predicted, class_data)
  return(list(cm, auc, df))
}

nnet_classifier(tr[c(1:14, 35)], ts[c(1:14)], ts[,35])



##LDA
set.seed(0)
ctrl <- rfeControl(functions = ldaFuncs,
                   method = 'cv',
                   number = 10,
                   verbose = FALSE)
lda_rfe <- rfe(train[, -35], train[, 35], 
               sizes = c(1:34),
               rfeControl = ctrl)
print(lda_rfe)
plot(lda_rfe, type = c('g', 'o'))

lda_rfe$optVariables

cols_lda <- lda_rfe$optVariables

j48_classifier(train[c(cols_lda, 'binaryClass')], test[cols_lda], test[,35])

nb_classifier(train[c(cols_lda, 'binaryClass')], test[cols_lda], test[,35])

logistic_classifier(train[c(cols_lda, 'binaryClass')], test[cols_lda], test[,35])

svm_classifier(train[c(cols_lda, 'binaryClass')], test[cols_lda], test[,35])

nnet_classifier(train[c(cols_lda, 'binaryClass')], test[cols_lda], test[,35])


#Random Forest
set.seed(0)
ctrl <- trainControl(method = "CV",
                     number = 10,
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)
rfFit <- train(x = train[, -35], 
               y = train[,35],
               method = "rf",
               ntree = 500,
               importance = TRUE,
               metric = "ROC",
               trControl = ctrl)
rfFit

## variable importance
imp <- varImp(rfFit)
imp

cols_rf <- rownames(imp$importance[order(imp$importance[1], decreasing = TRUE),])[1:10]

j48_classifier(train[c(cols_rf, 'binaryClass')], test[cols_rf], test[,35])

nb_classifier(train[c(cols_rf, 'binaryClass')], test[cols_rf], test[,35])

logistic_classifier(train[c(cols_rf, 'binaryClass')], test[cols_rf], test[,35])

svm_classifier(train[c(cols_rf, 'binaryClass')], test[cols_rf], test[,35])

nnet_classifier(train[c(cols_rf, 'binaryClass')], test[cols_rf], test[,35])


#XGBoost
set.seed(0)
ctrl <- trainControl(method = 'repeatedcv',
                     repeats = 1,
                     number = 10, 
                     verboseIter = TRUE,
                     search = 'random',
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary)
xgb_model <- train(binaryClass ~ ., data = train,
                   method = 'xgbTree',
                   metric = 'ROC',
                   maximize = TRUE,
                   trControl = ctrl)

importance <- xgb.importance(model = xgb_model$finalModel, 
               feature_names = colnames(train[,-35]))
cols_xgb <- importance[1:10, 'Feature']

j48_classifier(train[c(cols_xgb$Feature, 'binaryClass')], test[cols_xgb$Feature], test[,35])

nb_classifier(train[c(cols_xgb$Feature, 'binaryClass')], test[cols_xgb$Feature], test[,35])

logistic_classifier(train[c(cols_xgb$Feature, 'binaryClass')], test[cols_xgb$Feature], test[,35])

svm_classifier(train[c(cols_xgb$Feature, 'binaryClass')], test[cols_xgb$Feature], test[,35])

nnet_classifier(train[c(cols_xgb$Feature, 'binaryClass')], test[cols_xgb$Feature], test[,35])


#Lasso
set.seed(0)
y <- as.numeric(train[, 35])
lasso_model <- cv.glmnet(as.matrix(train[, -35]), y, alpha = 1, nfolds = 10)

lambda_opt <- lasso_model$lambda.min

lasso_fit <- glmnet(as.matrix(train[, -35]), y, alpha = 1, lambda = lambda_opt)

selected_Features <- as.numeric(coef(lasso_fit) != 0)[-1]

cols <- (1:ncol(train[,-35]))[selected_Features!=0]
cols <- colnames(train[, cols])

j48_classifier(train[c(cols, 'binaryClass')], test[cols], test[,35])

nb_classifier(train[c(cols, 'binaryClass')], test[cols], test[,35])

logistic_classifier(train[c(cols, 'binaryClass')], test[cols], test[,35])

svm_classifier(train[c(cols, 'binaryClass')], test[cols], test[,35])

nnet_classifier(train[c(cols, 'binaryClass')], test[cols], test[,35])

##when all features are considered, logistic regression model
logistic_classifier(train, test, test[,35])
