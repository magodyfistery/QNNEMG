function dataOut = smoothData(dataIn, W)
dataOut = zeros(size(dataIn));
numPoints = numel(dataIn);
beta = 1/W;
dataOut(1) = 0;
for i = 1:numPoints
    dataOut(i+1) = (1-beta)*dataOut(i) + beta*dataIn(i);
end
dataOut = dataOut(2:end);
% Bias Correction
den = 1 - beta.^(0:(numPoints-1));
dataOut = dataOut ./ den;
return