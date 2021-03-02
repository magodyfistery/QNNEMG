function initialTheta = initWeightsOptimiced(numNeuronsLayers)
% Randomly initialize the weights to small values

initialTheta = [];
for i = 2:length(numNeuronsLayers)   
    W = normrnd(0, 0.5, numNeuronsLayers(i), numNeuronsLayers(i - 1) + 1)/sqrt(numNeuronsLayers(i - 1));
    initialTheta = [initialTheta; W(:)];
end
return