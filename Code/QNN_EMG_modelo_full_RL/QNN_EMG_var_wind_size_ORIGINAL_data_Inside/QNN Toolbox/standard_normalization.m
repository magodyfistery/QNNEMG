function normaliced = standard_normalization(matrix)

matrix_mean = mean(matrix);
matrix_std = std(matrix);

if matrix_std == 0
    normaliced = matrix;
else
    normaliced = (matrix - matrix_mean)/matrix_std;
end

end
