library(DirichletReg)

dirichlet_model <- function(inputData_train, inputData_test, n_features) {
  
  if( ncol((inputData_train)) != ncol((inputData_test)) ) stop('Dataframes do not have the same shape.')
  
  data_train <- inputData_train[,1:n_features]
  data_train$Y <- DR_data (inputData_train[,(n_features+1):ncol(inputData_train)])
  
  data_test <- inputData_test[,1:n_features]
  data_test$Y <- DR_data (inputData_test[,(n_features+1):ncol(inputData_test)])
  
  res1 <- DirichReg(formula = Y ~ ., data_train)
  
  return(list(pred_train=predict(res1),pred_test=predict(res1,data_test)))
}
