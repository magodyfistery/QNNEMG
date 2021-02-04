function weights = trainQNN(numNeuronsLayers, transferFunctions, options, typeWorld, flagDisplayWorld)
% Window length for data smoothing
W = options. W;
% Initial settings
learningRate = options.learningRate;
numEpochs = options.numEpochs;
% Random initialization of the weights
theta = randInitializeWeights(numNeuronsLayers);
% Learning rate for training the ANN
alpha = options.learningRate;
% Epsilon
epsilon0 = options.epsilon;
epsilon = epsilon0;
if strcmp(options.typeUpdate, 'momentum') %  Momentum update
    % Setup for momentum update of the weights of the ANN
    momentum = options.initialMomentum;
    velocity = zeros( size(theta) );
    numEpochsToIncreaseMomentum = options.numEpochsToIncreaseMomentum;
else
    error('Invalid selection for updating the weights \n');
end
% Gamma
gamma = options.gamma;
% Training the ANN
cost = zeros(1, numEpochs);
% Initializing the history of the average reward
averageEpochReward = zeros(1, numEpochs);
cumulativeEpochReward = 0;
% Intializing the history of the average of the maxQ values
if strcmp(typeWorld, 'deterministic') || strcmp(typeWorld, 'randAgent')
    tw = 'randAgent';
    numStates = nchoosek(16, 4)*factorial(4);
elseif strcmp(typeWorld, 'randWorld')
    tw = 'randWorld';
    numStates = 13;
end
numTestingSamples = max(0.1*numStates, 20);
testStates = zeros(numTestingSamples, 64);
for i = 1:numTestingSamples
    stateAux = createWorld(tw);
    testStates(i, :) = stateAux(:)';
end
averageMaxQvalue = zeros(1, numEpochs);
% Maximum number of interactiobs allowed
maxIterationsAllowed = 100;
% History of wins
countWins = 0;
historyAverageWins = zeros(1, numEpochs);
for epoch = 1:numEpochs
    % Creating a new instance of the world
    state = createWorld(typeWorld);
    % Showing the initial state of the world;
    if flagDisplayWorld == true
        figure(1);
        displayWorld(state);
    end
    % Interaction of agent with the world
    gameOn = true; % Indicator of reaching a final state
    cumulativeGameReward = 0;
    numIteration = 0;
    dataX = zeros(maxIterationsAllowed, 64);
    dataY = zeros(maxIterationsAllowed, 4);
    while gameOn
        
        numIteration = numIteration + 1;
        % Reshaping the weights of the ANN
        weights = reshapeWeights(theta, numNeuronsLayers);
		% Predicting the response of the ANN for the current state
        [dummyVar, A] = forwardPropagation(state(:)', weights,...
        transferFunctions, options);
        Qval = A{end}(:, 2:end);
        [dummyVar, idx] = max(Qval);
        % Epsilon-greedy action selection
        if rand <= epsilon
			actionList = 1:4;
			actionList = actionList(actionList ~= idx);
            [dummyVar, idx] = sort( rand(1, 3) );
        end
        action = idx(1);
        % Taking the selected action
        if flagDisplayWorld == true
            displayAction(state, action);
        end
        % Getting the new state
        new_state = getState(state, action);
        % Getting the reward for the new state
        reward = getReward(new_state);
        if numIteration == maxIterationsAllowed
            reward = -10;
        end
        % Cumulative reward so far for the current episode
        cumulativeGameReward = cumulativeGameReward + reward;
        if flagDisplayWorld == true
            displayWorld(new_state);
        end
        % Q-Learning Algorithm
        % Getting the value of Q(s, a)
        [dummyVar, A] = forwardPropagation(state(:)', weights,...
            transferFunctions, options);
        old_Qval = A{end}(:, 2:end);
        % Getting the value of max_a'{Q(s', a')}
        [dummyVar, A] = forwardPropagation(new_state(:)', weights,...
            transferFunctions, options);
        new_Qval = A{end}(:, 2:end);
        maxQval = max(new_Qval);
        % Computation of the target
        if abs(reward) ~= 10
            % Target for a non-terminal state
            target = reward + gamma*maxQval;
        else
            % Taget for a terminal state
            target = reward;
        end
        if reward == -10 % end the game
            resultGame = 'lost';
            gameOn = false;
        elseif reward == +10
            resultGame = 'won ';
            countWins = countWins + 1;
            gameOn = false;
        end
        % Data for training the ANN
        dataX(numIteration, :) = state(:)';
        dataY(numIteration, :) = old_Qval;
        dataY(numIteration, action) = target;
        % Updating the state
        state = new_state;
    end
    dataXN = dataX(1:numIteration, :);
    dataYN = dataY(1:numIteration, :);
    % Updating the weights
    [cost(epoch), gradient] = regressionNNCostFunction(dataXN, dataYN,...
        numNeuronsLayers,...
        theta,...
        transferFunctions,...
        options);
    % Updating the weights of the ANN
    if strcmp(options.typeUpdate, 'momentum') %  Momentum update
        % Increase momentum after momIncrease iterations
        if epoch == numEpochsToIncreaseMomentum
            momentum = options.momentum;
        end
        velocity = momentum*velocity + alpha*gradient;
        theta = theta - velocity;
        % Annealing the learning rate
        alpha = learningRate*exp(-5*epoch/numEpochs);
    else
        error('Invalid selection for updating the weights \n');
    end
    % Cumulative reward of all the past episodes
    cumulativeEpochReward = cumulativeEpochReward + cumulativeGameReward;
    % History of the rewards for each episode
    averageEpochReward(epoch) = cumulativeEpochReward/epoch;
    % History of the average maxQval_er value for the test set of states
    [dummyVar, A] = forwardPropagation(testStates, weights, transferFunctions, options);
    test_Qval = A{end}(:, 2:end);
    averageMaxQvalue(epoch) = mean( max(test_Qval, [], 2) );
    % History of average of wins per epoch
    historyAverageWins(epoch) = countWins/epoch;
    fprintf('Epoch: %d of %d, cost = %3.2f, result = %s, epsilon = %1.2f, Qval = [%3.2f, %3.2f, %3.2f, %3.2f] \n',...
        epoch, numEpochs, cost(epoch), resultGame, epsilon, old_Qval(1), old_Qval(2), old_Qval(3), old_Qval(4));
    % Annealing the epsilon
    if epsilon > 0.10
        epsilon = epsilon0*exp(-7*epoch/numEpochs);
    end
end
% Plotting the cost function of each epoch
figure;
plot(1:epoch, cost(1:epoch), 'r', 'Linewidth', 1);
hold all;
costSmoothed = smoothData(cost(1:epoch), W);
plot(1:epoch, costSmoothed, 'b', 'Linewidth', 1);
hold off;
legend('Actual values', 'Smoothed values');
xlabel('epoch');
ylabel('cost');
title('Training cost vs. epochs');
grid on;
drawnow;
% Plotting the average reward of each episode
figure;
plot(1:epoch, averageEpochReward, 'b', 'Linewidth', 1);
hold all;
averageEpochRewardSmoothed = smoothData(averageEpochReward, W);
plot(1:epoch, averageEpochRewardSmoothed, 'c', 'Linewidth', 1);
hold off;
legend('Actual values', 'Smoothed values');
xlabel('epoch');
ylabel('Average reward per episode');
title('Average reward per episode vs epochs');
grid on;
drawnow;
% Plotting the average maxQ value for a test set of states
figure;
plot(1:epoch, averageMaxQvalue, 'm', 'Linewidth', 1);
hold all;
averageMaxQvalueSmoothed = smoothData(averageMaxQvalue, W);
plot(1:epoch, averageMaxQvalueSmoothed, 'r', 'Linewidth', 1);
hold off;
legend('Actual values', 'Smoothed values');
xlabel('epoch');
ylabel('Average maxQ value');
title('Average maxQ value vs epochs');
grid on;
drawnow;
% Plotting the average of wins per epoch
figure;
plot(1:epoch, historyAverageWins, 'c', 'Linewidth', 1);
hold all;
historyAverageWinsSmoothed = smoothData(historyAverageWins, W);
plot(1:epoch, historyAverageWinsSmoothed, 'b', 'Linewidth', 1);
hold off;
legend('Actual values', 'Smoothed values');
xlabel('epoch');
ylabel('Average of wins per epoch');
title('Average of wins per epoch vs epochs');
grid on;
drawnow;
% Reshaping the weights
weights = reshapeWeights(theta, numNeuronsLayers);
end