function [training_accuracy, test_accuracy] = QNN_emg_Exp_Replay(...
    params, emg_window_size, emg_stride, model_name, verbose_level)
%{
qnnOption: QNNOption {
    params.typeWorld;
    params.neurons_hidden1;
    params.neurons_hidden2;
    params.lambda;
    params.learning_rate;
    params.numEpochsToIncreaseMomentum;
    params.miniBatchSize;
    params.lambda: double;
    params.W: Int;  % Window length for data smoothing
    params.gamma; 
    params.epsilon;
}
verbose_level: int -> 0=no show messages, 1=show messages from first script
    level, 2=show messages from first to second script level...
%}

% Paths to other codes
addpath('Multivariate Regression Neural Network Toolbox');
addpath('Visualization Toolbox');
addpath('QNN Toolbox');
addpath('Gridworld Toolbox');
addpath(genpath('FeatureExtraction'));
addpath(genpath('Data'));
addpath(genpath('PreProcessing'));
addpath(genpath('testingJSON'));
addpath(genpath('trainingJSON'));
addpath('QNN Toolbox/model');
addpath(genpath('utils'));

%Conversion de JSON a .mat (si es necesario)
root_        = pwd;
data_gtr_dir = horzcat(root_,'\Data\General\training');
data_gts_dir = horzcat(root_,'\Data\General\testing');
data_sts_dir = horzcat(root_,'\Data\Specific');

if length(dir(data_gtr_dir))>2 || length(dir(data_gts_dir))>2 || length(dir(data_sts_dir))>2
    % No Data conversion
    disp('Data conversion already done');
else
    % Data conversion needed
    jsontomat;
end

% Variables for storing the results
assignin('base','s2', model_name);    
s = "QNN_Trained_Model_" + model_name + ".mat";
    
% parameters for data/signals of users
RepTraining = 1;       % uo to 125 numero de muestras que voy a usar en el entrenamiento x cada usuario en la carpeta C:\Users\juanp\Desktop\QNN - EMG - RandomData - Copy\Data\Specific
rangeDown=26;  %limite inferior de rango de muestras a leer
assignin('base','rangeDown', rangeDown); 

assignin('base','WindowsSize',  emg_window_size);
assignin('base','Stride',  emg_stride);
on  = true;
off = false;

%==============Parameters for Code_0 (preprocesser of emg data)==========

assignin('base','post_processing',     on);   %on si quiero post procesamiento en vector de etiquetas resultadnte                                          %off si quiero solo recomp -10 x recog 
assignin('base','RepTraining',  RepTraining); 
% if randomGestures is on, all will be random and packet EMG will not
% put the gestures one after other secuentially
assignin('base','randomGestures',     off);   %on si quiero leer datos randomicamente
assignin('base','noGestureDetection', off);  %off si no quiero considerar muestras con nogesture - OJO> actualmente el gesto predicho es la moda sin incluir no gesto
%limite superior de rango de muestras a leer
assignin('base','rangeValues', 150);  %up to 300 - rango de muestras PERMITIDO que uso dentro del dataset, del cual tomo "RepTraining" muestras
% if true: locates secuentially the gestures like: 
%   (nogestures if actived), fist, open, pinch, wave in, wave out
%   (nogestures if actived), fist, open, pinch, wave in, wave out, etc
assignin('base','packetEMG',     on); 

% Code_0 preprocess the emg data and correct, choosing the order of the
% gestures to be read too. Also, generates other relevant variables
Code_0(rangeDown);


% Parameters
typeWorld = 'randWorld';  % Type of the world of the game: deterministic, randAgent, and randWord
numNeuronsLayers = [40, params.neurons_hidden1, params.neurons_hidden2, 6];
transferFunctions = {'none', 'relu', 'relu', 'purelin'};


qnnOption = QNNOption(params.typeWorld, numNeuronsLayers, transferFunctions, ...
                params.lambda, params.learningRate, params.numEpochsToIncreaseMomentum, ...
                params.momentum, params.initialMomentum, ...
                params.miniBatchSize, params.W, params.gamma, params.epsilon);

qnn = QNN(qnnOption, params.rewardType, RepTraining);



tStart = tic;
[weights, summary] = qnn.train(verbose_level-1);
elapsedTimeHours = toc(tStart)/3600;
fprintf('\n Elapsed time for training: %3.3f h \n', elapsedTimeHours);

model_dir = strcat("results/models/", s);
save(model_dir,'weights', 'numNeuronsLayers',...
    'transferFunctions', 'options', 'typeWorld', 'flagDisplayWorld');


training_accuracy = summary(2)/(summary(2) + summary(5)) + summary(3)/(summary(3) + summary(6));

tStart = tic;
test_accuracy = qnn.test(139, verbose_level-1);
elapsedTimeHours = toc(tStart)/3600;
fprintf('Elapsed time for testing: %3.3f h \n', elapsedTimeHours);
end

