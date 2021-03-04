function initialTheta = initWeightsOptimiced(numNeuronsLayers)
% Randomly initialize the weights to small values

initialTheta = [];
for i = 2:length(numNeuronsLayers)   
    % W = normrnd(0, 1, numNeuronsLayers(i), numNeuronsLayers(i - 1) + 1)/sqrt(numNeuronsLayers(i - 1));
    W = rand(numNeuronsLayers(i), numNeuronsLayers(i - 1) + 1) * sqrt(1/numNeuronsLayers(i - 1));
    initialTheta = [initialTheta; W(:)];
end
initialTheta = normalize(initialTheta(:), 'zscore', 'std') * 0.01;

return