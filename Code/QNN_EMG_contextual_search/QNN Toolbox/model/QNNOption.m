classdef QNNOption < handle
    
    properties
        % default values
        typeWorld = 'randWorld';  % Type of the world of the game: deterministic, randAgent, and randWord
        numNeuronsLayers = [40, 80, 30, 6];
        transferFunctions = {'none', 'sigmoid', 'sigmoid', 'sigmoid'};
        reluThresh = 0;  % ????  %% ITS NOT PARAMETICED
        lambda = 0;  % regularization term
        
        % Momentum update
        learningRate = 0.1;
        typeUpdate = 'momentum'; %% ITS NOT PARAMETICED
        numEpochsToIncreaseMomentum = 50; %1000
        miniBatchSize = 25;
        momentum = 0.9;
        initialMomentum = 0.3;
        
        % Window length for data smoothing
        W = 25;                   
        % Q-learning settings
        gamma = 1; % Q-learning parameter
        epsilon = 1.00; %Initial value of epsilon for the epsilon-greedy exploration
        
        
        
        
    end
    
    methods
        
        
        function obj = QNNOption(typeWorld, numNeuronsLayers, transferFunctions, ...
                lambda, learningRate, numEpochsToIncreaseMomentum, ...
                momentum, initialMomentum, ...
                miniBatchSize, W, gamma, epsilon)
            % QNNOPTION Construct an instance of this class
            obj.typeWorld = typeWorld;
            obj.numNeuronsLayers = numNeuronsLayers;
            obj.transferFunctions = transferFunctions;
            % obj.reluThresh = reluThresh;
            obj.lambda = lambda;
            obj.learningRate = learningRate;
            % obj.typeUpdate = 'momentum';
            obj.numEpochsToIncreaseMomentum = numEpochsToIncreaseMomentum;
            obj.momentum = momentum;
            obj.initialMomentum = initialMomentum;            
            
            obj.miniBatchSize = miniBatchSize;
            obj.W = W;
            obj.gamma = gamma;
            obj.epsilon = epsilon;
            
        end
        
        
    end
end

