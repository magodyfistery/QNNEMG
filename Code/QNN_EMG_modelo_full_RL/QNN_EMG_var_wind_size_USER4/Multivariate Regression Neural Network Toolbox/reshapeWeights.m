function weights = reshapeWeights(theta, numNeuronsLayers)
% Reshaping the weight matrices
numLayers = length(numNeuronsLayers);
endPoint = 0;
weights = cell(1, numLayers - 1);
for i = 2:numLayers
    numRows = numNeuronsLayers(i);
    numCols = numNeuronsLayers(i - 1) + 1;
    numWeights = numRows*numCols;
    startPoint = endPoint + 1;
    endPoint = endPoint + numWeights;
    weights{i - 1} = reshape( theta(startPoint:endPoint), numRows, numCols );
end
return