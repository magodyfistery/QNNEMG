function J = nnCostFunction(neurons_per_layer, thetas_rolled, X, y, lambda)
  % X es solo un ejemplo
  
  %X
  %y
  %thetas_rolled
  
  m = size(X, 1);
  number_layers = size(neurons_per_layer, 2);
  
  
  
  %%% Feed Forward
  index_reshape_begin = 1;
  index_reshape_end = 0;
  indexes_reshapes_theta = zeros(number_layers-1, 2);
  
  
  a = [ones(m, 1) X];
  
  
  for i=1:number_layers-1
    index_reshape_end = index_reshape_end + (neurons_per_layer(i)+1)*neurons_per_layer(i+1);
    theta = reshape(thetas_rolled(index_reshape_begin:index_reshape_end), (neurons_per_layer(i)+1), neurons_per_layer(i+1));
    
    z = a * theta;
    a = [ones(m, 1) sigmoid(z)];
    indexes_reshapes_theta(i, :) = [index_reshape_begin index_reshape_end];
    index_reshape_begin = index_reshape_end + 1;
  end
  
  % activations_functions
  % activations_derivates_functions
  % indexes_activations_functions
  
  % hipotesis es a, la respuesta correcta es y (hot encoded)
  sparse_y = sparse_one_hot_encoding(y, neurons_per_layer(number_layers));

  % sum(m x num_labels) => 1 x num_labels; sum(1 x num_labels) => 1x1
  regularization = 0; % (lambda/(2*m)) * (sum(sum( Theta1(:, 2:end) .^ 2 )) + sum(sum( Theta2(:, 2:end) .^ 2 )));

  J = sum(sum( -sparse_y .* log(a(:, 2:end)) - (1 - sparse_y) .* log(1 - a(:, 2:end))))/m + regularization;
  
    
end
