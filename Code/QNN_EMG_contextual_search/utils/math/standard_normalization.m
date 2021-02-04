function normaliced = standard_normalization(matrix)

matrix_mean = mean(matrix);
matrix_std = std(matrix);

normaliced = (matrix - matrix_mean)/matrix_std;
end
