grad_data <- read.csv("data.csv", sep = ";")

set.seed(1234)

ind = sample(1:nrow(grad_data), size = (nrow(grad_data) * 0.8), replace = FALSE)
grad_data$Target <- as.factor(grad_data$Target)
train = grad_data[ind, ]
test = grad_data[-ind, ]

library(rpart)
library(rpart.plot)

# Fit the decision tree
dat_rpart <- rpart(Target ~ . , data = train, method = 'class')
par(mar=rep(0.5, 4))
plot(dat_rpart)
text(dat_rpart)
prp(dat_rpart)
rpart.plot(dat_rpart)

plotcp(dat_rpart)
dat_rpart$cptable

dat_min_cp = dat_rpart$cptable[which.min(dat_rpart$cptable[,"xerror"]),"CP"]
dat_min_cp

dat_pruned <- prune(dat_rpart, cp = dat_min_cp)
rpart.plot(dat_pruned)

library(randomForest)
p <- floor((ncol(train))/ 2)

rf_model <- randomForest(as.factor(Target) ~ ., data = train, 
                         mtry = p, 
                         importance = TRUE)
rf_model
importance(rf_model)
varImpPlot(rf_model)

library(glmnet)

# Use cross-validation to select lambda
cv_model <- cv.glmnet(x = as.matrix(train[, -37]),
                      y = as.factor(train$Target),
                      family = "multinomial")

# Best lambda from CV
best_lambda <- cv_model$lambda.1se

# Predict using the best lambda
preds <- predict(cv_model, 
                 newx = as.matrix(test[, -37]), 
                 type = "class", 
                 s = best_lambda)

# This should now be a vector
conf_matrix <- table(Predicted = preds, Actual = test$Target)
conf_matrix

# Load the package
library(class)

# Prepare training and testing data
x_train <- as.matrix(train[, -37])
y_train <- as.factor(train$Target)

x_test <- as.matrix(test[, -37])
y_test <- test$Target

# Fit kNN (k = 5 is common, but you can tune this)
knn_preds <- knn(train = x_train, test = x_test, cl = y_train, k = 5)

# Confusion matrix
conf_matrix_knn <- table(Predicted = knn_preds, Actual = y_test)
conf_matrix_knn

library(caret)

# Try k from 1 to 20 and choose the best one
k_grid <- expand.grid(k = 1:20)

ctrl <- trainControl(method = "cv", number = 5)

knn_tuned <- train(x = x_train,
                   y = y_train,
                   method = "knn",
                   trControl = ctrl,
                   tuneGrid = k_grid)
plot(knn_tuned$results$k, 1-knn_tuned$results$Accuracy, xlab = "K", 
     ylab = "Classification Error", type="b", pch = 19, col = "darkblue")

# Best k
knn_tuned$bestTune

final_model <- train(
  as.factor(Target) ~ ., 
  data = train, 
  method = "knn", 
  tuneGrid = data.frame(k = knn_tuned$bestTune$k)
)

# Predict with best k
knn_preds_best <- predict(final_model, newdata = test)

# Confusion matrix
conf_matrix_best <- table(Predicted = knn_preds_best, Actual = y_test)
conf_matrix_best

confusionMatrix(knn_preds_best, as.factor(test$Target))
confusionMatrix(knn_preds, as.factor(test$Target))
confusionMatrix(as.factor(preds), as.factor(test$Target))

tree_model <- rpart(as.factor(Target) ~ . , data = train, method = "class")
tree_preds <- predict(dat_pruned, newdata = test, type = "class")
confusionMatrix(tree_preds, as.factor(test$Target))
rpart.plot(tree_model)


prop.table(table(grad_data$Target))
prop.table(table(train$Target))
prop.table(table(test$Target))

rf_preds <- predict(rf_model, newdata = test)
confusionMatrix(rf_preds, test$Target)



