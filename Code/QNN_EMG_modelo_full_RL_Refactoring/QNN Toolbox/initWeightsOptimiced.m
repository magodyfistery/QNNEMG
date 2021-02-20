function initialTheta = initWeightsOptimiced(numNeuronsLayers)
% Randomly initialize the weights to small values
initialTheta = [];
for i = 2:length(numNeuronsLayers)   
    W = normrnd(0, 0.5, numNeuronsLayers(i), numNeuronsLayers(i - 1) + 1)* sqrt(2/(numNeuronsLayers(i - 1) + numNeuronsLayers(i)));
    initialTheta = [initialTheta; W(:)];
end
return