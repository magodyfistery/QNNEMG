function prediction = predict(neurons_per_layer, thetas_rolled, X)
  
   
  m = size(X, 1);
  number_layers = size(neurons_per_layer, 2);
  
  
  
  %%% Feed Forward
  index_reshape_begin = 1;
  index_reshape_end = 0;
  
  
  a = [ones(m, 1) X];
  
  
  
  for i=1:number_layers-1
    index_reshape_end = index_reshape_end + ((neurons_per_layer(i)+1)*neurons_per_layer(i+1));
    theta = reshape(thetas_rolled(index_reshape_begin:index_reshape_end), (neurons_per_layer(i)+1), neurons_per_layer(i+1));
    
    z = a * theta;
    a = [ones(m, 1) sigmoid(z)];
    index_reshape_begin = index_reshape_end + 1;
  end
  
  [value, indexs] = max(a(:, 2:end), [], 2);
  prediction = indexs;
    
  
end
