# Course Project - Practical Machine Learning
url1 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
url2 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

training <- read.csv(url1)
validation <- read.csv(url2)

library(caret)

#### Feature selection
# erase first 7 columns as these variables (name, time, window etc.)
# are not relevant for our prediction
training <- subset(training, select = -c(1:7))
validation <- subset(validation, select = -c(1:7))

# search for near zero variance variables and omit them
nzv <- nearZeroVar(training, saveMetrics = TRUE, names = TRUE)
nzvIndex <- which(nzv$nzv == TRUE)
training <- subset(training, select = -nzvIndex)
validation <- subset(validation, select = -nzvIndex)
# check if it worked
nearZeroVar(training)
nearZeroVar(validation)

# now remove variables with too many nas
colNA <- colSums(is.na(training))
unique(colNA)
# as we can see, there are only columns with either
# 0 NA values or 19216 Nas.
# Lets remove those with the NAs
training <- training[,  colNA == 0]
validation <- validation[, colNA == 0]

# now lets take a look at the correlation between the remaining variables
cor_mat <- cor(subset(training, select = -c(classe)))
# too many variables, lets try the findCorrelation function from the caret package
colCor <- findCorrelation(cor_mat, cutoff = 0.8)

# remove those columns from data set
training <- training[, -colCor]
validation <- validation[, -colCor]

# Split training set into training and test set. 
# The original test set will be our validation set.
inTrain <- createDataPartition(training$classe, p = 0.7, list = FALSE)
testing <- training[-inTrain,]
training <- training[inTrain,]


# Lets try some model fitting
# first LDA:
mod_lda <- train(classe ~ ., method = "lda", data = training)
pred_lda <- predict(mod_lda, newdata = testing)
confusionMatrix(pred_lda, testing$classe)$overall[1]
# accuracy for test set only 0.636


# now boosting (gbm) - TAKES TOO LONG
#mod_gbm <- train(classe ~ ., method = "gbm", data = training, verbose = FALSE)

# decision tree
mod_tree <- train(classe ~ ., method = "rpart", data = training)
pred_tree <- predict(mod_tree, newdata = testing)
confusionMatrix(pred_tree, testing$classe)$overall[1]
# poor accuracy (0.538) as well


# try support vector machine
library(e1071)
mod_svm <- svm(classe ~ ., data = training)
#mod_svm
# check accuracy on test set
pred_svm <- predict(mod_svm, newdata = testing)
confusionMatrix(pred_svm, testing$classe)$overall[1]
# pretty good accuracy 0.937

# use svm model for validation set
pred_svm_validation <- predict(mod_svm, newdata = validation)
pred_svm_validation

# try bagging poor fitting models (lda and tree)
# first lda
# save predictors and outcome separately
predictors <- training[, c(1:length(training)-1)]
outcome <- training$classe
mod_lda_bag <- bag(predictors, outcome, B = 10,
                   bagControl = bagControl(fit = ldaBag$fit,
                                           predict = ldaBag$pred,
                                           aggregate = ldaBag$aggregate))

mod_lda_bag

pred_lda_bag <- predict(mod_lda_bag, newdata = testing)
confusionMatrix(pred_lda_bag, testing$classe)$overall[1]
# no significant improvement

# now try bagged decision tree
mod_tree_bag <- bag(predictors, outcome, B = 10,
                    bagControl = bagControl(fit = ctreeBag$fit,
                                            predict = ctreeBag$pred,
                                            aggregate = ctreeBag$aggregate))

pred_tree_bag <- predict(mod_tree_bag, newdata = testing)
confusionMatrix(pred_tree_bag, testing$classe)$overall[1]
# accuracy has improved a lot: 0.95
# lets use this model to predict the validation set

# class of columns differ between training and validation. 
# Lets fix this
classes_val <- sapply(validation, class)
classes_train <- sapply(training, class)

# find columns that have different types
result <- data.frame(classes_val, classes_train, stringsAsFactors = FALSE)
result[!(result$classes_val == result$classes_train), ]

# change magnet_dumbbell_z, magnet_forearm_y and magnet_forearm_z
# to numeric type in validation set
validation$magnet_dumbbell_z <- as.numeric(validation$magnet_dumbbell_z)
validation$magnet_forearm_y <- as.numeric(validation$magnet_forearm_y)
validation$magnet_forearm_z <- as.numeric(validation$magnet_forearm_z)

pred_tree_bag_validation <- predict(mod_tree_bag, newdata = validation)
pred_tree_bag_validation

# compare pred_tree_bag_validation and pred_svm_validation
table(pred_svm_validation, pred_tree_bag_validation)
# exactly the same classification

