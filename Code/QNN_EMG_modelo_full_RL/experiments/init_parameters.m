function init_parameters()
global parameters_training
parameters_training = table2array(readtable('experiments/experiments_parameters_QNN1_TRAIN.csv'));
global parameters_validation
parameters_validation = table2array(readtable('experiments/experiments_parameters_QNN1_VALIDATION.csv'));
global parameters_testing
parameters_testing = table2array(readtable('experiments/experiments_parameters_QNN1_TESTING.csv'));
global verbose
verbose = false;  % if true, the program will print ALL and use CPU for that
global index_numNeuronsLayers_input
index_numNeuronsLayers_input = 2;
global index_numNeuronsLayers_output
index_numNeuronsLayers_output = 5;
global index_RepTraining
index_RepTraining = 6;
global index_SamplingType
index_SamplingType = 7;  % 0=Random package, NO USADO
global index_learningRate
index_learningRate = 8;
global index_numEpochsToIncreaseMomentum
index_numEpochsToIncreaseMomentum = 9;
global index_miniBatchSize
index_miniBatchSize = 10;
global index_lambda
index_lambda = 11;
global filename_experimentsQNN
filename_experimentsQNN = '../experiments/experimentsQNN/experiments.xlsx'; % relative
end

