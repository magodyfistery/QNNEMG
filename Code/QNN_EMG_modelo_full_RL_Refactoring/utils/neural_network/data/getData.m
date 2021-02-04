function [X, y] = getData(samples)

  if nargin == 0
      samples = -1;
  end
  
  load('image_data');
  m = size(X, 1);
  idx = randperm(m);
  
  X_shuffled = X(idx, :);
  y_shuffled = y(idx, :);
  
  if(samples <= 0)
    X = X_shuffled;
    y = y_shuffled;
  elseif(samples > 0)
    X = X_shuffled(1:samples,:);
    y = y_shuffled(1:samples,:);

  end
    
    
  
  
  end
