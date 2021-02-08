params.W = 25;   % Window length for data smoothing
params.typeWorld = 'randWorld'; % Type of the world of the game: deterministic, randAgent, and randWord
params.rewardType = 1;
repTrainingTrain = 88;
repTrainingTest = 25;
params.reserved_space_for_gesture = 10; % getNumberWindows(997, window_size, stride, false);
params.miniBatchSize = 32;% getNumberWindows(997, window_size, stride, false);
params.initialMomentum = 0.3;
params.momentum = 0.9;
params.numEpochsToIncreaseMomentum = ceil(repTrainingTrain/2);
params.gamma = 0.99; % Q-learning parameter

params.epsilon = 0.3; % epsilon-greedy exploration

window_size = 200;  % 250
stride = 20;% ceil(window_size/5);  % jump between windows

params.learningRate = 0.1;

params.numNeuronsLayers = [40, 40, 40, 6];
params.transferFunctions = {'none', 'relu', 'relu', 'purelin'};

params.lambda = 0.01;


addpath(genpath('utils'))


verbose_level = 2;
model_name = "just_testing";

[training_accuracy, test_accuracy, qnn] = QNN_emg_Exp_Replay(..., 
    params, window_size, stride, model_name, verbose_level, repTrainingTrain, repTrainingTest);

% fprintf("\n\nTraining pond. accuracy: %2.2f, test pond. accuracy: %2.2f\n", training_accuracy, test_accuracy);

theta = qnn.theta;
experience_replay = qnn.gameReplay;
total_num_windows_predicted = qnn.total_num_windows_predicted;
save("theta.mat",'theta');
save("experience_replay.mat",'experience_replay');
save("total_num_windows_predicted.mat",'total_num_windows_predicted');
