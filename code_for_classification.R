library("tidymodels")
library("tidyverse")
library("bonsai")
library("finetune")
library("imputeTS")
library("splines")
library("ranger")
library("caret")
library("jsonlite")

# Data reading
atrain <- read.csv("C:/Users/garbersa/Desktop/Data Challenge/aortaP_train_data.csv")
btrain <- read.csv("C:/Users/garbersa/Desktop/Data Challenge/brachP_train_data.csv")
atest <- read.csv("C:/Users/garbersa/Desktop/Data Challenge/aortaP_test_data.csv")
btest <- read.csv("C:/Users/garbersa/Desktop/Data Challenge/brachP_test_data.csv")

glimpse(atrain)

# Interpolation

a <- t(atrain)
anew <- na_interpolation(a[2:(nrow(a)-1),])
anew <- t(anew)

b <- t(btrain)
bnew <- na_interpolation(b[2:(nrow(b)-1),])
bnew <- t(bnew)

# Data Cleaning

#anew <- anew %>% select(-target)
datatotal <- cbind(anew, bnew,atrain$target) %>% as_tibble()
#datatotal[,673] #targets



# bsplines

Xa.spline = bs(1:dim(anew)[2], df = 8) 
features.spline.a = apply(anew, 1, function(x) lm(x ~ Xa.spline - 1)$coef)
features.spline.a = t(features.spline.a)


Xb.spline = bs(1:dim(bnew)[2], df = 8) 
features.spline.b = apply(bnew, 1, function(x) lm(x ~ Xb.spline - 1)$coef)
features.spline.b = t(features.spline.b)

features.spline = as_tibble(cbind(features.spline.a, features.spline.b, target = factor(atrain$target)))
features.spline <- features.spline %>% mutate(
  target = factor(target)
) 



features.spline = as_tibble(features.spline)
cores <- parallel::detectCores(logical = FALSE)

set.seed(1)
split <- initial_split(features.spline, strata = target)
data_train <- training(split)
data_test <- testing(split)

rf_model <- rand_forest(
  mode = "classification",
  mtry = 5,
  min_n = 17,
  trees = 1000
) %>%
  set_engine("ranger", num.threads = cores, importance = "impurity")

rec_final <- recipe(target ~., data = data_train) %>%
  step_normalize(all_predictors()) %>%
  #step_corr(threshold = 0.8)%>%
  step_zv(all_predictors())

wf <-workflow() %>%
  add_recipe(rec_final) %>%
  add_model(rf_model)

rf_fit <- wf %>% fit(data_train)

predictions <- predict(rf_fit, new_data = data_test[,-dim(data_test)[2]])
confusionMatrix(predictions$.pred_class, data_test$target)

# Interpolation

a.test <- t(atest)
anew.test <- na_interpolation(a.test[2:(nrow(a.test)),])
anew.test <- t(anew.test)

b.test <- t(btest)
bnew.test <- na_interpolation(b.test[2:(nrow(b.test)),])
bnew.test <- t(bnew.test)

data_test_wolabels <- cbind(anew.test,bnew.test) %>% as_tibble()


Xa.spline = bs(1:dim(anew.test)[2], df = 8) 
features.spline.a = apply(anew.test, 1, function(x) lm(x ~ Xa.spline - 1)$coef)
features.spline.a = t(features.spline.a)


Xb.spline = bs(1:dim(bnew.test)[2], df = 8) 
features.spline.b = apply(bnew.test, 1, function(x) lm(x ~ Xb.spline - 1)$coef)
features.spline.b = t(features.spline.b)

features.spline = as_tibble(cbind(features.spline.a, features.spline.b))


predictions <- predict(rf_fit, new_data = features.spline)

predictions %>% ggplot(aes(.pred_class)) + geom_bar()

predictions %>% write.csv( "C:/Users/garbersa/Desktop/predictions.csv")

tib <- cbind(seq.int(0,874),predictions)

exportJSON <- toJSON(tib)
write(exportJSON, "C:/Users/garbersa/Desktop/predictions.json")
