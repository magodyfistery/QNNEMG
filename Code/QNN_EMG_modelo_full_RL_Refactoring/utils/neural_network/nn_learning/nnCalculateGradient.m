function gradient = nnCalculateGradient(neurons_per_layer, thetas_rolled, X, y, lambda)

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
  
  
  a = [ones(1,1) X];
  
  activations_functions = zeros(sum(neurons_per_layer)+number_layers-1, 1);
  activations_derivates_functions = zeros(sum(neurons_per_layer)+number_layers-1, 1);
  
  indexes_activations_functions = zeros(number_layers, 2);  % 2 por ser inicio y fin
  index_activations_begin = 1;
  index_activations_end = neurons_per_layer(1)+1;
  
  activations_functions(index_activations_begin:index_activations_end, 1) = a(:);
  indexes_activations_functions(1, :) = [index_activations_begin index_activations_end];
  
  index_activations_begin = index_activations_end+1;
  
  
  
  for i=1:number_layers-1
    index_reshape_end = index_reshape_end + (neurons_per_layer(i)+1)*neurons_per_layer(i+1);
    theta = reshape(thetas_rolled(index_reshape_begin:index_reshape_end), (neurons_per_layer(i)+1), neurons_per_layer(i+1));
    
    
    z = a * theta;
    a = [ones(1,1) sigmoid(z)];
    a_derivate = [ones(1,1) sigmoidGradient(z)];
    
    index_activations_end = index_activations_end + neurons_per_layer(i+1)+1;
    
    activations_functions(index_activations_begin:index_activations_end, 1) = a(:);
    activations_derivates_functions(index_activations_begin:index_activations_end, 1) = a_derivate(:);
    
    indexes_reshapes_theta(i, :) = [index_reshape_begin index_reshape_end];
    indexes_activations_functions(i+1, :) = [index_activations_begin index_activations_end];
    
    
    
    index_reshape_begin = index_reshape_end + 1;
    index_activations_begin = index_activations_end+1;
  end
  
  % hipotesis es a, la respuesta correcta es y (hot encoded)
  sparse_y = sparse_one_hot_encoding(y, neurons_per_layer(number_layers));

  % sum(m x num_labels) => 1 x num_labels; sum(1 x num_labels) => 1x1
  % regularization = 0; % (lambda/(2*m)) * (sum(sum( Theta1(:, 2:end) .^ 2 )) + sum(sum( Theta2(:, 2:end) .^ 2 )));

  % J = sum(sum( -sparse_y .* log(a(2:end)) - (1 - sparse_y) .* log(1 - a(2:end))))/m + regularization;
  gradient = zeros(size(thetas_rolled));
  
  % Backpropagation
  
  ind = indexes_activations_functions(number_layers, :);
  % la ultima capa de activación no requiere el uno agregado del bias
  h_t = activations_functions(ind(1):ind(2));
  h = h_t(2:end)';  % ignora el bias
  delta = h - sparse_y;
  
  i=number_layers-1;
  
  while i >= 1
    ind = indexes_activations_functions(i, :);
    ind_theta = indexes_reshapes_theta(i, :);
    a = activations_functions(ind(1):ind(2))';  % se requiere el bias
    a_d = activations_derivates_functions(ind(1):ind(2));
    a_derivate = a_d(2:end)';
    theta_t = reshape(thetas_rolled(ind_theta(1):ind_theta(2)), neurons_per_layer(i)+1, neurons_per_layer(i+1));
    
    theta = theta_t(2:end,:)';
    
    
    
    % delta
    
    grad = zeros(size(theta, 1), size(theta, 2) + 1);
    grad = (grad + delta' * a)';
    
    gradient(ind_theta(1):ind_theta(2), 1) = grad(:);
    
    
    if i > 1
      delta = (delta * theta) .* a_derivate;
    end
    
    
    
    i = i - 1;
    
  end
  
  
  
    
end
