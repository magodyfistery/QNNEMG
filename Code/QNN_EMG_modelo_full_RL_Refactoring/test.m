addpath(genpath('utils'))

model_name = "just_testing";
verbose_level = 2;

repTrainingTrain = 88;
repTrainingTest = 25;

window_size = 200;
stride = 20;

% NN parameters
params.numNeuronsLayers = [40, 40, 40, 6];
params.transferFunctions = {'none', 'relu', 'relu', 'purelin'};
params.initialMomentum = 0.3;
params.momentum = 0.9;
params.numEpochsToIncreaseMomentum = ceil(repTrainingTrain/2);
params.learningRate = 0.1;
params.lambda = 0.01;

% Q-learning parameter
params.rewardType = 1;
params.gamma = 0.99; 
params.reserved_space_for_gesture = 10; % getNumberWindows(997, window_size, stride, false);
params.miniBatchSize = 32;
params.epsilon = 0.3; % epsilon-greedy exploration


[training_accuracy, test_accuracy, qnn] = QNN_emg_Exp_Replay(..., 
    params, window_size, stride, model_name, verbose_level, repTrainingTrain, repTrainingTest);


