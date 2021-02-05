params.learningRate = 0.1;
params.neurons_hidden1 = 50;
params.neurons_hidden2 = 50;
params.numEpochsToIncreaseMomentum = 50;
params.lambda = 0;
params.momentum = 0.9;
params.initialMomentum = 0.3;
% Q-learning settings
params.gamma = 0.9; % Q-learning parameter
params.epsilon = 0.1; %Initial value of epsilon for the epsilon-greedy exploration
params.W = 25;   % Window length for data smoothing
params.typeWorld = 'randWorld'; % Type of the world of the game: deterministic, randAgent, and randWord

params.rewardType = 1;

window_size = 230;  % 250
stride = 20;% ceil(window_size/5);  % jump between windows


params.reserved_space_for_gesture = ceil(getNumberWindows(997, window_size, stride, false)/3);

params.miniBatchSize = getNumberWindows(997, window_size, stride, false);

verbose_level = 2;
model_name = "just_testing";

[training_accuracy, test_accuracy, qnn] = QNN_emg_Exp_Replay(..., 
    params, window_size, stride, model_name, verbose_level, 15, 5);

% fprintf("\n\nTraining pond. accuracy: %2.2f, test pond. accuracy: %2.2f\n", training_accuracy, test_accuracy);

theta = qnn.theta;
experience_replay = qnn.gameReplay;
total_num_windows_predicted = qnn.total_num_windows_predicted;
save("theta.mat",'theta');
save("experience_replay.mat",'experience_replay');
save("total_num_windows_predicted.mat",'total_num_windows_predicted');
