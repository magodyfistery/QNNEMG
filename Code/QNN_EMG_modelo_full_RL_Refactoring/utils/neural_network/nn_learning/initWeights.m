function initial_thetas_rolled = initWeights(neurons_per_layer, lower_limit, upper_limit)
  
  sum_total_weights = 0;
  for i=1:size(neurons_per_layer, 2)-1
    sum_total_weights = sum_total_weights + (neurons_per_layer(i)+1) * neurons_per_layer(i+1);
    
  end
  
  random_matrix = lower_limit + rand(sum_total_weights, 1) * upper_limit * 2;
  initial_thetas_rolled = standard_normalization(random_matrix);
  
  
end
