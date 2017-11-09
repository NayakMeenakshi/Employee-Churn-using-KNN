library(caret)
library(onehot)
library(pROC)
set.seed(1234)
data <- read.csv("E:/Applied Machine Learning/Project2_Meenakshi/HR_comma_sep.csv", 
                 header=TRUE)
head(data)
typeof(data$left)
#One HotEncoding
data_ <- onehot(data,stringsAsFactors = FALSE,addNA = FALSE,max_levels = 10)
pr <- as.data.frame(predict(data_,data))
head(pr)

#Scaling data to [0,1] range
maxs <- apply(pr, 2, max)    
mins <- apply(pr, 2, min)
scaled <- as.data.frame(scale(pr, center = mins, scale = maxs - mins))
scaled$left <- factor(scaled$left, labels = c( "remained","left"))
head(scaled)

# splitting data into train, test 
set.seed(1234)
index <- createDataPartition(scaled$left, p = .70,list = FALSE, times = 1)
train_knn <- scaled[ index,]
test_knn <- scaled[-index,]
head(test_knn)
#Multiple split for getting different training data sizes
train_split = list(scaled[sample(nrow(scaled), nrow(scaled)*.2),],
                   scaled[sample(nrow(scaled), nrow(scaled)*.3),],
                   scaled[sample(nrow(scaled), nrow(scaled)*.4),],
                   scaled[sample(nrow(scaled), nrow(scaled)*.5),],
                   scaled[sample(nrow(scaled), nrow(scaled)*.6),],
                   scaled[sample(nrow(scaled), nrow(scaled)*.7),])
#Store training data size in %
x_ax = c(20,30,40,50,60,70)

#KNN model using caret
set.seed(3331)
head(train_knn)
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

knn_fit <- train(left ~., data = train_knn, method = "knn",
                 trControl=trctrl,
                 preProcess = c("center", "scale"),
                 tuneLength = 15)
knn_fit
train_accuracy_knn <- knn_fit$results$Accuracy
train_error_knn <- 1-knn_fit$results$Accuracy
#training error for best fit K
1-knn_fit$results$Accuracy[knn_fit$results$k == knn_fit$bestTune$k]

#plot train error versus K
plot(train_error_knn,  main='Training Error vs K',xlab="K",ylab='Training Error Rate',col='blue')
knnPred <- predict(knn_fit,newdata = test_knn)
confusionMatrix(knnPred, test_knn$left)
test_error_knn = 1 -mean(knnPred == test_knn$left)
test_error_knn

##ROC curve
auc_knn_test <- roc(as.numeric(test_knn$left), as.numeric(knnPred))
print(auc_knn_test)
plot(auc_knn_test, print.thres=TRUE, col = 'blue',main = 'ROC curve for HR data')

#KNN training and test error for multiple training data size
set.seed(3333)
train_error_split <- c()
test_error_split <- c()
for (i in train_split)
{
  knnfit <- train(left ~., data = i, method = "knn",tuneLength = 15)
  knnPred <- predict(knnfit,newdata = i)
  error<-1 -mean(knnPred == i$left)
  train_error_split <- c(train_error_split,error)
  TestPred <- predict(knnfit,newdata = test_knn)
  errortest<-1 -mean(TestPred == test_knn$left)
  test_error_split <- c(test_error_split,errortest)
  print(train_error_split)
  print(test_error_split)
}
train_error_split
test_error_split

#Learning Curves
#Plot training error vs training data size
plot(x_ax,train_error_split,  main='Training Error vs Training data size',
     xlab="Training data size",ylab='Train Error',col='red',type='l')

#Plot test error vs training data size
plot(x_ax,test_error_split,  main='Test Error vs Training data size',
     xlab="Training data size",ylab='Test Error',col='red',type='l')

#PLOT BOTH THE TRAIN AND TEST ERROR CURVES WITHIN THE SAME PLOT
plot(x_ax, train_error_split, ylim=c(0.044,0.077), type="l", col="green", ylab="error rate", 
     xlab="training data size(in %)", main="Train and Test Error")
lines(x_ax,test_error_split,col="red")


