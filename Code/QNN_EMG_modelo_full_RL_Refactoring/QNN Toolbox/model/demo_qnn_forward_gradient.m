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

window_size = 230;
stride = 30;  % jump between windows

verbose_level = 2;
model_name = "just_testing";

% [training_accuracy, test_accuracy] = QNN_emg_Exp_Replay(params, window_size, stride, model_name, verbose_level);



addpath(genpath('utils'));
addpath(genpath('QNN Toolbox'));

[X, y] = getData(1000);
[X_train, y_train, X_test, y_test, X_validation, y_validation] = split_train_test_validation(X, y, 0.7, 0.1);


numNeuronsLayers = [400 30 30 10];
transferFunctions = {'none', 'sigmoid', 'sigmoid', 'sigmoid'};

qnnOption = QNNOption(params.typeWorld, numNeuronsLayers, transferFunctions, ...
                params.lambda, params.learningRate, params.numEpochsToIncreaseMomentum, ...
                params.momentum, params.initialMomentum, ...
                params.miniBatchSize, params.W, params.gamma, params.epsilon);

            
qnn = QNN(qnnOption, params.rewardType, -1);
qnn.initTheta(initWeights(qnn.qnnOption.numNeuronsLayers, -1, 1))

numEpochs = 10;
total_costs = zeros(1, numEpochs);
total_costs_validation = zeros(1, numEpochs);

for epoch=1:numEpochs
    fprintf("epoch %d of %d\n", epoch, numEpochs);
    cost = nnCostFunction(qnn.qnnOption.numNeuronsLayers, qnn.theta, X_train, y_train, 0);
    cost_validation = nnCostFunction(qnn.qnnOption.numNeuronsLayers, qnn.theta, X_validation, y_validation, 0);
    total_costs(1, epoch) = cost;
    total_costs_validation(1, epoch) = cost_validation;
    for i=1:size(X_train, 1)
        gradient = qnn.calculateGradientForOneObservation(X_train(i,:), y_train(i,:));
        qnn.theta = qnn.theta - (qnn.qnnOption.learningRate * gradient);
    end
end

figure(1)
hold on;
plot(1:numel(total_costs), total_costs); 
plot(1:numel(total_costs_validation), total_costs_validation); 


[Js_train, Js_validation] = learningCurves(qnn.qnnOption.numNeuronsLayers, qnn.theta, X_train, y_train, X_validation, y_validation, qnn.qnnOption.learningRate, qnn.qnnOption.lambda);

figure(2)
hold on;
plot(1:size(Js_train, 2), Js_train, 'b');
plot(1:size(Js_validation, 2), Js_validation, 'r');
xlabel('m')
ylabel('J')
legend('training', 'validation')
title('Learning Curves')

prediction = predict(qnn.qnnOption.numNeuronsLayers, qnn.theta, X_test);
accuracy = sum(prediction == y_test)/numel(y_test)



