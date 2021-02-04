function [X_train y_train X_test y_test X_validation y_validation] = split_train_test_validation(X, y, train_portion, validation_portion)
  
  test_portion = 1 - train_portion - validation_portion;
  
  m = size(X, 1);
  idx = randperm(m);
  
  X_shuffled = X(idx, :);
  y_shuffled = y(idx, :);
  
  m_train = ceil(m * train_portion);
  X_train = X_shuffled(1:m_train, :);
  y_train = y_shuffled(1:m_train, :);
  
  
  
  m_test = ceil(m * test_portion); 
 
  X_test = X_shuffled(m_train+1:m_test+m_train, :);
  y_test = y_shuffled(m_train+1:m_test+m_train, :);
  
  X_validation = X_shuffled(m_test+m_train+1:end, :); 
  y_validation = y_shuffled(m_test+m_train+1:end, :);
  
end
