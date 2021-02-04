function [costValue, gradientValues] = regressionNNCostFunction(dataX, dataY,...
    numNeuronsLayers,...
    theta,...
    transferFunctions,...
    metaParameters)
% This function computes the cost and gradient of a feed-forward neural network
% for non-linear multivariate regression
%
% Inputs:
% dataX                   [N n] matrix, where each row contains an observation
%                         X = (x_1, x_2,...,x_n)
%
% dataY                   [N numOutputs] vector, where each row contains
%                         the actual values of the outputs
%
% numNeuronsLayers        [1 L] vector [#_1, #_2,..., #_L], where #_1
%                         denotes the size of the input layer, #_2 denotes
%                         the size of the first hidden layer, #_3 denotes
%                         the size of the second hidden layer, and so on, and
%                         #_L = 1 denotes the size of the output layer
%
% theta                   Vector that contains all the weights of the
%                         neural network
%
% transferFunctions       Cell containg the name of the transfer functions
%                         of each layer of the neural network. Options of transfer
%                         functions are:
%                         - none: input layer has no transfer functions
%                         - tanh: hyperbolic tangent
%                         - elu: exponential linear unit
%                         - softplus: log(exp(x) + 1)
%                         - relu: rectified linear unit
%                         - logsig: logistic function
%                         - purelin: transfer function of the output layer
%
% metaParameters         structure containing additional settings for the
%                        neural network (e.g., rectified linear unit
%                        threshold, lambda, number of iterations, etc.)
%
% Outputs
% costValue               Contains the cost value of the function to train
%                         the neural network
% gradientValues          Vector containing the gradients of all the
%                         weights of the neural network
%
% Escuela Politecnica Nacional
% Marco E. Benalcázar Palacios
% marco.benalcazar@epn.edu.ec

% Regularization parameters
lambda = metaParameters.lambda;

% Number of training examples
N = size(dataX, 1);

% Number of output units
C = size(dataY, 2);

% Number of layers of the neural network
numLayers = length(numNeuronsLayers);

% Threshold for the RELU transfer function
reluThresh = metaParameters.reluThresh;

% Reshaping the weight matrices
endPoint = 0;
totalNumberWeights = 0;
W = cell(1, numLayers - 1);
for i = 2:numLayers
    numRows = numNeuronsLayers(i);
    numCols = numNeuronsLayers(i - 1) + 1;
    numWeights = numRows*numCols;
    startPoint = endPoint + 1;
    endPoint = endPoint + numWeights;
    W{i - 1} = reshape( theta(startPoint:endPoint), numRows, numCols );
    totalNumberWeights = totalNumberWeights + numWeights;
end

%% Forward propagation
[Z, A] = forwardPropagation(dataX, W, transferFunctions, metaParameters);
yHat = A{end}(:, 2:end); % [yHat_1(X) yHat_2(X) ... yHat_c(X)]
% [  ...         ...     ...     ...   ]
% [yHat_1(X) yHat_2(X) ... yHat_c(X)]

%% Cost function
% Computing the value of the cost function given the weights of the network
kte = (1/N)*(1/C);
mse = kte*(1/2)*sum( sum( (yHat - dataY).^2 ) ); % MS error
% Computing the value of the regularization function
regularization = 0;
for i = 2:numLayers
    regularization = regularization + sum(  sum( W{i - 1}(:,2:end).^2 )  );
end
% Computing the total cost = cost of errors + regularization
costValue = mse + kte*lambda*(1/2)*regularization;

%% Back propagation algorithm
delta{numLayers} = kte*(yHat - dataY);
vectorOnes = ones(N, 1);
Wgradient = W; % Intializing the gradient matrices
for i = (numLayers - 1):-1:1
    % Computing the gradients of the neurons of the i-th layer based on the
    % delta of the (i + 1) layer and the output of the i-th layer
    if i == (numLayers - 1)
        switch transferFunctions{i + 1}
            case 'logsig'
                df_dz = logsig(Z{i + 1}).*(1 - logsig(Z{i + 1}));
            case 'relu'
                df_dz = Z{i + 1} > reluThresh;
            case 'tanh'
                df_dz = 1 - tansig(Z{i + 1}).^2;
            case 'sigmoid'
                df_dz = sigmoidGradient(Z{i + 1});
            case 'purelin'
                df_dz = ones( size(Z{i + 1}) );
            case 'softplus'
                df_dz = logsig(Z{i});
            case 'elu'
                df_dz = (Z{i + 1} > 0) + (Z{i + 1} <= 0).*(elu(Z{i + 1}) + 1);
            otherwise
                error('Invalid transfer function. Valid options are elu, softplus, relu, logsig, tanh, and purelin');
        end
        delta{i + 1} = delta{i + 1}.*df_dz;
        Wgradient{i} = delta{i + 1}'*A{i};
    else
        Wgradient{i} = delta{i + 1}(:,2:end)'*A{i};
    end
    % Adding the gradient of the regularization term
    Wgradient{i}(:,2:end) = Wgradient{i}(:,2:end) + kte*lambda*W{i}(:,2:end);
    if i == 1
        break; % if i == 1 there are no more gradients to compute so the back-propagation
        % algorithm must end
    end
    % Computing the derivatives of the transfer functions
    switch transferFunctions{i}
        case 'logsig'
            transferFcnDerivative = logsig(Z{i}).*(1 - logsig(Z{i}));
        case 'relu'
            transferFcnDerivative = Z{i} > reluThresh;
        case 'tanh'
            transferFcnDerivative = 1 - tansig(Z{i}).^2;
        case 'purelin'
            transferFcnDerivative = ones( size(Z{i}) );
        case 'softplus'
            transferFcnDerivative = logsig(Z{i});
        case 'elu'
            transferFcnDerivative = (Z{i} > 0) + (Z{i} <= 0).*(elu(Z{i}) + 1);
        otherwise
            error('Invalid transfer function. Valid options are elu, softplus, relu, logsig, tanh, and purelin');
    end
    if i < (numLayers - 1)
        delta{i} = ( delta{i + 1}(:,2:end)*W{i} ).*[vectorOnes transferFcnDerivative];
    else
        delta{i} = ( delta{i + 1}*W{i} ).*[vectorOnes transferFcnDerivative];
    end
end

% Vectorizing all the weights of the neural network (column vector)
gradientValues = zeros(totalNumberWeights,1);
endPoint = 0;
for i = 2:numLayers
    numRows = numNeuronsLayers(i);
    numCols = numNeuronsLayers(i - 1) + 1;
    numWeights = numRows*numCols;
    startPoint = endPoint + 1;
    endPoint = endPoint + numWeights;
    gradientValues(startPoint:endPoint,1) = Wgradient{i - 1}(:);
end
return