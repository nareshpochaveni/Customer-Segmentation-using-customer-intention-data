#importing all the required libraries
install.packages("data.table")
install.packages("caret")
install.packages("GGally")
install.packages("lattice")
install.packages("corrplot")
install.packages("rattle")
install.packages("gbm")
library(data.table)
library(caret)
library(lattice)
library(GGally)
library(ggplot2)
library(corrplot)
library(rattle)
library(gbm)

# loading the dataset from the current location
data = fread("online_shoppers_intention.csv")

# check dataset description and summary 
str(data)
head(data)
summary(data)

# check for missing values
is.na((data))

# restructure the data
# changing column type of logical into int then as facotr
data$Revenue <- ifelse(data$Revenue ==TRUE,1,0)
data$Revenue = as.factor(data$Revenue)
data$Weekend <- ifelse(data$Weekend ==TRUE,1,0)
data$Weekend = as.factor(data$Weekend)

#Performed encoding of variables by chaning catagorical variables into factors
data$Browser = as.factor(data$Browser)
data$OperatingSystems  = as.factor(data$OperatingSystems)
data$Region  = as.factor(data$Region)
data$TrafficType = as.factor(data$TrafficType)
data$VisitorType = as.factor(data$VisitorType)

# Exploratory Data Analysis

#plotting the correlation graph to check correlation between different features
correlation <- cor(data[,c(1:10)]) 
corrplot(correlation, method = "shade", type = "lower",diag = FALSE)

# plotting relation between Bounce rate and exit rate 
ggplot(data = data,aes(x = BounceRates, y = ExitRates))+ 
  geom_point(mapping = aes(color = Revenue)) + 
  geom_smooth(se = TRUE, alpha = 0.5,color="blue",method="lm")+theme()

# bar graphs for Revenue 
ggplot(data, aes(x = Revenue,fill=Revenue))+geom_bar()
# bar graphs Revenue vs Visitor Type
ggplot(data, aes(x =VisitorType,fill=Revenue))+geom_bar()
# bar graphs for TrafficType
ggplot(data, aes(x = TrafficType,fill=Revenue))+geom_bar()
# bar graphs for Region
ggplot(data, aes(x = Region,fill=Revenue))+geom_bar(fill=rainbow(9))
# bar graphs for Special Day
ggplot(data, aes(x = SpecialDay,fill=Revenue))+geom_bar(fill=rainbow(6))
# violin plot
ggplot(data, aes(x = ProductRelated,y=Revenue))+geom_violin(trim = FALSE)

# set seed 
set.seed(12345678)

# saved preprocessed data
write.csv(data,"Prepocesed.csv", row.names = FALSE)

#Model training and testing
#splitting the data into training and testing set
training = createDataPartition(y = data$Revenue, p = 0.75, list = FALSE)
train_set = data[training, ]
test_set = data[-training, ]

#Logistic Regression
model_logistic_reg = glm(data = train_set, Revenue~ PageValues + ProductRelated_Duration + ProductRelated + ExitRates + Administrative_Duration + BounceRates + Administrative + Informational_Duration, family = binomial)
df_test_set$predicted_property_probability = predict(model_logistic_reg,newdata = df_test_set,type='response')
df_test_set$predicted_property = ifelse(df_test_set$predicted_property_probability >=0.50,1,0)
confusionMatrix = table(df_test_set$Revenue,df_test_set$predicted_property)

#CART model
model_fitting = train(data = train_set, Revenue~ PageValues + ProductRelated_Duration + ProductRelated + ExitRates + Administrative_Duration + BounceRates + Administrative + Informational_Duration, method = "rpart")
model_fitting$finalModel
plot(model_fitting$finalModel)
text(model_fitting$finalModel)
fancyRpartPlot(model_fitting$finalModel)

confusionMatrix(predict(model_fitting, newdata = test_set), test_set$Revenue)
confusionMatrix(predict(model_fitting, newdata = train_set), train_set$Revenue)


#DECISION TREE DIAGRAM(IF NEEDED)

#install.packages("rpart.plot")
library(rpart)
library(rpart.plot)
fit <- rpart(Revenue~., data = train_set, method = 'class')
rpart.plot(fit, extra = 106)


#Bagging model

model_fitting_bagging = train(data = train_set, Revenue~ PageValues + ProductRelated_Duration + ProductRelated + ExitRates + Administrative_Duration + BounceRates + Administrative + Informational_Duration, method = "treebag")

model_fitting_bagging

confusionMatrix(predict(model_fitting_bagging, newdata = test_set), test_set$Revenue)
confusionMatrix(predict(model_fitting_bagging, newdata = train_set), train_set$Revenue)

#Boosting model

model_fitting_boosting = train(data = train_set, Revenue~ PageValues + ProductRelated_Duration + ProductRelated + ExitRates + Administrative_Duration + BounceRates + Administrative + Informational_Duration, method = "gbm", verbose = FALSE)

model_fitting_boosting

confusionMatrix(predict(model_fitting_boosting, newdata = train_set), train_set$Revenue)
confusionMatrix(predict(model_fitting_boosting, newdata = test_set), test_set$Revenue)

#Random Forest model

model_fitting_rf = train(data = train_set, Revenue~ PageValues + ProductRelated_Duration + ProductRelated + ExitRates + Administrative_Duration + BounceRates + Administrative + Informational_Duration, method = "rf",prox = TRUE)

model_fitting_rf

confusionMatrix(predict(model_fitting_rf, newdata = train_set), train_set$Revenue)
confusionMatrix(predict(model_fitting_rf, newdata = test_set), test_set$Revenue)

#K-nn model

model_fitting_knn = train(data = train_set, Revenue~ PageValues + ProductRelated_Duration + ProductRelated + ExitRates + Administrative_Duration + BounceRates + Administrative + Informational_Duration, method = "knn")

model_fitting_knn

confusionMatrix(predict(model_fitting_knn, newdata = train_set), train_set$Revenue)
confusionMatrix(predict(model_fitting_knn, newdata = test_set), test_set$Revenue)

#Naive Bayes model

model_fitting_nb = train(data = train_set, Revenue~ PageValues + ProductRelated_Duration + ProductRelated + ExitRates + Administrative_Duration + BounceRates + Administrative + Informational_Duration, method = "nb")

model_fitting_nb

confusionMatrix(predict(model_fitting_nb, newdata = train_set), train_set$Revenue)
confusionMatrix(predict(model_fitting_nb, newdata = test_set), test_set$Revenue)

#plotting variable importance graph for various models

varImp(model_fitting_bagging)
varImp(model_fitting_boosting)

#Plotting the ROC curve for the boosting model
#install.packages("ROCR")
library("ROCR")
true_pred <- predict(model_fitting_boosting, newdata = test_set, type = "prob")[,2]
Pred2 = prediction(true_pred, test_set$Revenue)
plot(performance(Pred2, "tpr", "fpr"))
abline(0, 1, lty = 2)
