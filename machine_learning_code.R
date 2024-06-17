# 2016 TICS ML

## XGB Prep

suppressPackageStartupMessages({
  library(tidyverse)
  library(gbm)
  library(caret)
  library(xgboost)
  library(pls)
  library(glmnet)
})

seed <- 1234

set.seed(seed)

parts <- caret::createDataPartition(tempdat$cog_4scales_16, p = .5, list = F)
train <-  tempdat[parts, ]
test <-  tempdat[-parts, ]

X_train <- train %>% 
  select(!c(cog_4scales_16, rahhidpn)) %>% 
  data.matrix()
Y_train <- train[,'cog_4scales_16']

X_test <- test %>% 
  select(!c(cog_4scales_16, rahhidpn)) 
Y_test <- test[,'cog_4scales_16']
id_test <- test[,'rahhidpn']

xgb_train = xgb.DMatrix(data = data.matrix(X_train), label = unlist(Y_train))
xgb_test = xgb.DMatrix(data = data.matrix(X_test), label = unlist(Y_test))

## XGB Modeling

set.seed(seed)

best_param = list()
best_logloss = Inf
best_logloss_index = 0
iter = 0
a <- Sys.time()

for (i in 1:5) {
  for (j in 1:5) {
    for (k in 1:3) {
      b <- Sys.time()
      iter <- iter + 1
      print(paste0('Running iteration ', iter, ' / ', (5*5*3)))
      print(paste0('depth = ', c(1:5)[i], 
                   '    eta = ', c(0.001, 0.1, 0.3, 0.7, 1)[j],
                   '    min_child_weight = ', c(1, 5, 15)[k]
      ))
      param <- list(objective = "reg:squarederror",
                    max_depth = c(1:5)[i],
                    eta = c(0.001, 0.1, 0.3, 0.7, 1)[j],
                    min_child_weight = c(1, 5, 15)[k]
                    # gamma = 0
      )
      cv.nround = 100000
      cv.nfold = 5
      set.seed(seed)
      mdcv <- xgb.cv(data = xgb_train, params = param, nthread = 3, 
                     nfold = 5, nrounds = cv.nround,
                     verbose = F, early_stopping_rounds = 20, maximize = F)
      
      min_logloss = min(mdcv$evaluation_log[, test_rmse_mean])
      print(paste0('Minimum RMSE this iteration :',min_logloss))
      min_logloss_index = which.min(mdcv$evaluation_log[, test_rmse_mean])
      print(paste0('Iter selected this iteration :',min_logloss_index))
      print(mdcv)
      
      if (min_logloss < best_logloss) {
        best_logloss = min_logloss
        best_logloss_index = min_logloss_index
        best_param = param
      }
      print(paste0('Total time: ', Sys.time() - a))
      print(paste0('Time this round: ', Sys.time() - b))
      # }
    }}}
beepr::beep(3)
rm(a, b)

print(best_param)
print(best_logloss_index)

nround = best_logloss_index

################################################################################

set.seed(seed)

final <- xgb.train(data=xgb_train, params=best_param, nrounds=nround, nthread=2)

final_tics_all <- xgb.train(data=xgb_train, params=best_param, nrounds=nround, nthread=2)

pred_y <- predict(final, xgb_test)

print(xgb.ggplot.importance(xgb.importance(model = final)))
print('R-squared')
print(summary(lm(Y_test$cog_4scales_16 ~ pred_y))$r.square)
print(cor(Y_test$cog_4scales_16, pred_y))
print(ModelMetrics::rmse(lm(Y_test$cog_4scales_16 ~ pred_y)))






# 2020 Langa-Weir CIND or Dementia Onset

### XGB Prep

suppressPackageStartupMessages({
  library(tidyverse)
  library(gbm)
  library(caret)
  library(xgboost)
  library(pls)
  library(glmnet)
})

seed <- 1234

set.seed(seed)

parts <- caret::createDataPartition(tempdat$cind_dem_onset_18_20,
                                    p = .5, list = F)
train <-  tempdat[parts, ]
test <-  tempdat[-parts, ]

X_train <- train %>% 
  select(!c(cind_dem_onset_18_20, rahhidpn)) %>% 
  data.matrix()
Y_train <- train[,'cind_dem_onset_18_20']

X_test <- test %>% 
  select(!c(cind_dem_onset_18_20, rahhidpn))  
Y_test <- test[,'cind_dem_onset_18_20']
id_test <- test[,'rahhidpn']

xgb_train = xgb.DMatrix(data = data.matrix(X_train), label = unlist(Y_train))
xgb_test = xgb.DMatrix(data = data.matrix(X_test), label = unlist(Y_test))
```

### XGB Modeling

set.seed(seed)

best_param = list()
best_logloss = Inf
best_logloss_index = 0
iter = 0
a <- Sys.time()

for (i in 1:5) {
  for (j in 1:5) {
    for (k in 1:3) {
      b <- Sys.time()
      iter <- iter + 1
      print(paste0('Running iteration ', iter, ' / ', (5*5*3)))
      print(paste0('depth = ', c(1:5)[i], 
                   '    eta = ', c(0.001, 0.1, 0.3, 0.7, 1)[j],
                   '    min_child_weight = ', c(1, 5, 15)[k]
      ))
      param <- list(objective = "binary:logistic",
                    max_depth = c(1:5)[i],
                    eta = c(0.001, 0.1, 0.3, 0.7, 1)[j],
                    min_child_weight = c(1, 5, 15)[k]
                    # gamma = 0
      )
      cv.nround = 100000
      cv.nfold = 5
      set.seed(seed)
      mdcv <- xgb.cv(data = xgb_train, params = param, nthread = 3, 
                     nfold = 5, nrounds = cv.nround,
                     verbose = F, early_stopping_rounds = 20, maximize = F)
      
      min_logloss = min(mdcv$evaluation_log[, test_logloss_mean])
      print(paste0('Minimum RMSE this iteration :',min_logloss))
      min_logloss_index = which.min(mdcv$evaluation_log[, test_logloss_mean])
      print(paste0('Iter selected this iteration :',min_logloss_index))
      print(mdcv)
      
      if (min_logloss < best_logloss) {
        best_logloss = min_logloss
        best_logloss_index = min_logloss_index
        best_param = param
      }
      print(paste0('Total time: ', Sys.time() - a))
      print(paste0('Time this round: ', Sys.time() - b))
      # }
    }}}

print(best_param)
print(best_logloss_index)

nround = best_logloss_index

################################################################################

set.seed(seed)


final <- xgb.train(data=xgb_train, params=best_param, nrounds=nround, nthread=2)
final_lw_all <- xgb.train(data=xgb_train, params=best_param, nrounds=nround, 
                          nthread=2)

pred_y <- predict(final, xgb_test)

print(xgb.ggplot.importance(xgb.importance(model = final)))
print('AUC')
pROC::roc(Y_test$cind_dem_onset_18_20, pred_y, plot = T)
cor(Y_test$cind_dem_onset_18_20, pred_y, method = 'spearman')
summary(glm(Y_test$cind_dem_onset_18_20 ~ pred_y, family = 'binomial'))