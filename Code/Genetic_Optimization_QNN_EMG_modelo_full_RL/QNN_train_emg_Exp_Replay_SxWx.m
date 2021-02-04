function [training_accuracy, test_accuracy] = QNN_train_emg_Exp_Replay_SxWx(...
    params, verbose)
warning off all;

% Type of the world of the game: deterministic, randAgent, and randWord
typeWorld = 'randWorld';

% Creating an artificial neural network for Q-learning
addpath('Multivariate Regression Neural Network Toolbox');
addpath('Visualization Toolbox');
addpath('QNN Toolbox');
addpath('Gridworld Toolbox');
addpath(genpath('FeatureExtraction'));
addpath(genpath('Data'));
addpath(genpath('PreProcessing'));
addpath(genpath('testingJSON'));
addpath(genpath('trainingJSON'));

% Architecture of the artificial neural network
%%%%PARAMETRIZACION_DANNY%%%%
numNeuronsLayers = [40, ceil(params.neurons_hidden1), ceil(params.neurons_hidden2), 6];              %CAMBIAR [40, 75, 50, 6]; - #clases  [64, 75, 50, 6]
transferFunctions{1} = 'none';
transferFunctions{2} = 'relu';
transferFunctions{3} = 'relu';
transferFunctions{4} = 'purelin';
options.reluThresh = 0;
%%%%PARAMETRIZACION_DANNY%%%%
options.lambda = params.lambda; % CAMBIAR regularization term
% SGD and online learning settings
%Code_0;
%dataPacketSize1   = evalin('base', 'dataPacketSize'); %numero de epocas ahora depende de numero de datos de usuarios en la carpeta
%options.numEpochs = dataPacketSize1; %dataPacketSize; %100000  %REVISAR

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


%if noGestureDetection==on tengo que considerar que el valore de
%RepTraining debe ser rangeValues-25 si rangeValues<150 o tambien
%rangeValues-50 si rangeValues<300. Esto es xq los nogestos ahora no se
%cuentan, y no puedo poner 150 reps si no se tomaran en cuenta las 25 del
%no gesto.
%# determinado de cada clase x muestra, cambiar

%%%%PARAMETRIZACION_DANNY%%%%
window_size = params.window_size;
%%%%PARAMETRIZACION_DANNY%%%%
assignin('base','WindowsSize',  window_size);   %CAMBIAR - 200
stride = params.stride;


%%%%PARAMETRIZACION_DANNY%%%%
assignin('base','Stride',  stride);        %CAMBIAR - 20

%%%%PARAMETRIZACION_DANNY%%%%
RepTraining = 100;       % uo to 125 numero de muestras que voy a usar en el entrenamiento x cada usuario en la carpeta C:\Users\juanp\Desktop\QNN - EMG - RandomData - Copy\Data\Specific
on  = true;
off = false;
assignin('base','Reward_type',     on);   %on si quiero recompensa -1 x ventana (clasif) y -10 x recog
assignin('base','post_processing',     on);   %on si quiero post procesamiento en vector de etiquetas resultadnte                                          %off si quiero solo recomp -10 x recog 
assignin('base','RepTraining',  RepTraining); 
assignin('base','randomGestures',     off);   %on si quiero leer datos randomicamente
assignin('base','noGestureDetection', off);  %off si no quiero considerar muestras con nogesture - OJO> actualmente el gesto predicho es la moda sin incluir no gesto
%limite superior de rango de muestras a leer
assignin('base','rangeValues', 150);  %up to 300 - rango de muestras PERMITIDO que uso dentro del dataset, del cual tomo "RepTraining" muestras
assignin('base','packetEMG',     on);

rangeDown=26;  %limite inferior de rango de muestras a leer
assignin('base','rangeDown', rangeDown); 
Code_0(rangeDown);
dataPacketSize1     = evalin('base', 'dataPacketSize');
options.numEpochs  = RepTraining*(dataPacketSize1-2);  %numero total de muestras de todos los usuarios

% Momentum update
%%%%PARAMETRIZACION_DANNY%%%%
options.learningRate = params.learning_rate;            % 9e-3  CAMBIAR
options.typeUpdate = 'momentum';
options.momentum = 0.9;
options.initialMomentum = 0.3;

%%%%PARAMETRIZACION_DANNY%%%%
options.numEpochsToIncreaseMomentum = params.numEpochsToIncreaseMomentum; %1000

s2 = "genetic_train";

%%%%PARAMETRIZACION_DANNY%%%%                   %CAMBIAR # de EXPERIMENTO
options.miniBatchSize = params.miniBatchSize; %CAMBIAR % Size of the minibatch from experience replay

% Window length for data smoothing
options. W = 25;                   

% Q-learning settings
options.gamma = 1; % Q-learning parameter
options.epsilon = 1.00; %Initial value of epsilon for the epsilon-greedy exploration
% Fitting the ANN
flagDisplayWorld = false;
tStart = tic;

s1 = 'QNN_Trained_Model_';
assignin('base','s2', s2);    
s3 = '.mat';
s = strcat(s1,s2,s3);

%%%%PARAMETRIZACION_DANNY Y CAMBIO EN trainQNN (verbose) y summary%%%%
[weights, summary] = trainQNN_Exp_Replay(numNeuronsLayers, transferFunctions, options, typeWorld, flagDisplayWorld, verbose);

elapsedTimeHours = toc(tStart)/3600;
fprintf('\n Elapsed time for training: %3.3f h \n', elapsedTimeHours);
save('QNN_Trained_Model.mat','weights', 'numNeuronsLayers',...
    'transferFunctions', 'options', 'typeWorld', 'flagDisplayWorld');

%%%%CAMBIODANNYINICIO%%%%
model_dir = strcat("results/models/", s);
save(model_dir,'weights', 'numNeuronsLayers',...
    'transferFunctions', 'options', 'typeWorld', 'flagDisplayWorld');

%%%%CAMBIODANNYFIN%%%%
recognition_accuracy = summary(1)/(summary(1) + summary(4));
clasification_accuracy = summary(2)/(summary(2) + summary(5));
win_per_window_accuracy = summary(3)/(summary(3) + summary(6));
training_accuracy = recognition_accuracy + clasification_accuracy + win_per_window_accuracy;

test_accuracy = QNN_TESTING_Exp_Replay(126, params, verbose);
end

