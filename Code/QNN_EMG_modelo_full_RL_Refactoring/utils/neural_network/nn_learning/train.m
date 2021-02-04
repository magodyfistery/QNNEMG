function [new_thetas_rolled, Js_train, Js_validation] = ...
    train(neurons_per_layer, thetas_rolled, epochs, ...
    X_train, y_train, X_validation, y_validation, alpha, lambda, verbose)

  Js_train = [];
  Js_validation = [];
  
  new_thetas_rolled = thetas_rolled(:);

  debug_step = ceil(epochs/3);
  
  for epoch=1:epochs
    
    if mod(epoch, debug_step) == 0 && verbose
        fprintf('Epoch: %i\n', epoch);
    end
    
    J_train = nnCostFunction(neurons_per_layer, new_thetas_rolled, X_train, y_train, lambda);
    J_validation = nnCostFunction(neurons_per_layer, new_thetas_rolled, X_validation, y_validation, lambda);
    
    Js_train = [Js_train J_train];
    Js_validation = [Js_validation J_validation];
   
    for i=1:size(X_train, 1)
        gradient = nnCalculateGradient(neurons_per_layer, new_thetas_rolled, X_train(i, :), y_train(i, :), lambda);
        new_thetas_rolled = new_thetas_rolled - (alpha * gradient);
        
    end
  end
end
