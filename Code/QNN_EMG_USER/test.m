params.learningRate = 0.1;
params.neurons_hidden1 = 30;
params.neurons_hidden2 = 30;
params.numEpochsToIncreaseMomentum = 50;
params.miniBatchSize = 25;
params.lambda = 0;
params.momentum = 0.9;
params.initialMomentum = 0.3;
% Q-learning settings
params.gamma = 1; % Q-learning parameter
params.epsilon = 1.00; %Initial value of epsilon for the epsilon-greedy exploration
params.W = 25;   % Window length for data smoothing
params.typeWorld = 'randWorld'; % Type of the world of the game: deterministic, randAgent, and randWord

params.rewardType = 1;

window_size = 300;
stride = 200;  % jump between windows

verbose_level = 10;
model_name = "just_testing";

[training_accuracy, test_accuracy] = QNN_emg_Exp_Replay(params, window_size, stride, model_name, verbose_level);