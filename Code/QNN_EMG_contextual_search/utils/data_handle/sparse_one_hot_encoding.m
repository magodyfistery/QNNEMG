function one_hot_encoding = sparse_one_hot_encoding(X, number_clases)
% X - m x 1 vector of classes -> [1,2,3,...,6]
% sparse_one_hot_encoding - m x number_clases

m = size(X, 1);
one_hot_encoding = zeros(m, number_clases);

for i=1:m
    class_number = X(i,1);
    if class_number > 0
        one_hot_encoding(i, class_number) = 1;
    elseif class_number < 0
        one_hot_encoding(i, :) = ones(1, number_clases);
        one_hot_encoding(i, -class_number) = 0;
    else
        disp('ERROR, the number of class should be > 0 or < 0');
    end
    
end
  
end
