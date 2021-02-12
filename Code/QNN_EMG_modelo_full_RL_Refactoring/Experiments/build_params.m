function params = build_params(row_params)
%BUILD_PARAMS Summary of this function goes here
%   Detailed explanation goes here

% other
params.window_size = row_params(5);
params.stride = row_params(6);

% NN parameters
params.numNeuronsLayers = [row_params(7), row_params(8), row_params(9), row_params(10)];

params.transferFunctions = {'none', ...
     castNumberToTransferFunction(row_params(11)),...
     castNumberToTransferFunction(row_params(12)),...
     castNumberToTransferFunction(row_params(13))};

params.initialMomentum = row_params(14);
params.momentum = row_params(15);
params.numEpochsToIncreaseMomentum = row_params(16);
params.learningRate = row_params(17);
params.lambda = row_params(18);

% Q-learning parameter
params.rewardType = row_params(19);
params.gamma = row_params(20); 
params.reserved_space_for_gesture = row_params(21); % getNumberWindows(997, window_size, stride, false);
params.miniBatchSize = row_params(22);
params.epsilon = row_params(23); % epsilon-greedy exploration

params.numEpochs = row_params(24);
end

