params.learningRate = 0.1;
params.neurons_hidden1 = 30;
params.neurons_hidden2 = 20;
params.numEpochsToIncreaseMomentum = 50;
params.miniBatchSize = 40;
params.reserved_space_for_gesture = 40;
params.lambda = 0;
params.momentum = 0.9;
params.initialMomentum = 0.3;
% Q-learning settings
params.gamma = 1; % Q-learning parameter
params.epsilon = 0.1; %Initial value of epsilon for the epsilon-greedy exploration
params.W = 25;   % Window length for data smoothing
params.typeWorld = 'randWorld'; % Type of the world of the game: deterministic, randAgent, and randWord

params.rewardType = 1;

window_size = 250;
stride = ceil(window_size/5);  % jump between windows

verbose_level = 2;
model_name = "just_testing";

[training_accuracy, test_accuracy, qnn] = QNN_emg_Exp_Replay(..., 
    params, window_size, stride, model_name, verbose_level, 87, 13);

% fprintf("\n\nTraining pond. accuracy: %2.2f, test pond. accuracy: %2.2f\n", training_accuracy, test_accuracy);

theta = qnn.theta;
experience_replay = qnn.gameReplay;
total_num_windows_predicted = qnn.total_num_windows_predicted;
save("theta.mat",'theta');
save("experience_replay.mat",'experience_replay');
save("total_num_windows_predicted.mat",'total_num_windows_predicted');
clear all;
