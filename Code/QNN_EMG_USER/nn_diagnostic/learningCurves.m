function [Js_train, Js_validation] = learningCurves(neurons_per_layer, thetas_rolled, X_train, y_train, X_validation, y_validation, alpha, lambda);
  
  % Number of training examples
  m = size(X_train, 1);

  % You need to return these values correctly
  Js_train = zeros(1, m);
  Js_validation   = zeros(1, m);



  for i=1:m
    
    fprintf('\nm=: %i\n', i);

    
    [new_thetas_rolled dummy1 dummy2] = train(neurons_per_layer, thetas_rolled, 1, X_train(1:i,:), y_train(1:i), X_validation, y_validation, alpha, lambda);
    
    J_train = nnCostFunction(neurons_per_layer, new_thetas_rolled, X_train(1:i,:), y_train(1:i), lambda);
    J_validation = nnCostFunction(neurons_per_layer, new_thetas_rolled, X_validation, y_validation, lambda);  % todo el dataset
    
    Js_train(1, i) = J_train;
    Js_validation(1, i) = J_validation;
    
  end
  
end
