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
RepTraining = 10;       % uo to 125 numero de muestras que voy a usar en el entrenamiento x cada usuario en la carpeta C:\Users\juanp\Desktop\QNN - EMG - RandomData - Copy\Data\Specific
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




% Parameters
typeWorld = 'randWorld';  % Type of the world of the game: deterministic, randAgent, and randWord
numNeuronsLayers = [40, params.neurons_hidden1, params.neurons_hidden2, 6];
transferFunctions = {'none', 'sigmoid', 'sigmoid', 'sigmoid'};


qnnOption = QNNOption(params.typeWorld, numNeuronsLayers, transferFunctions, ...
                params.lambda, params.learningRate, params.numEpochsToIncreaseMomentum, ...
                params.momentum, params.initialMomentum, ...
                params.miniBatchSize, params.W, params.gamma, params.epsilon);

            
qnn = QNN(qnnOption, params.rewardType, RepTraining);
qnn.initTheta(initWeights(qnn.qnnOption.numNeuronsLayers, -1, 1))



numEpochs = 1;

tStart = tic;

mean_acuracy_by_epoch = zeros(1, numEpochs);
recognition_accuracy = zeros(1, numEpochs);
classif_mode_ok_accuracy = zeros(1, numEpochs);
classif_by_window_accuracy = zeros(1, numEpochs);

for epoch=1:numEpochs
    % Code_0 preprocess the emg data and correct, choosing the order of the
    % gestures to be read too. Also, generates other relevant variables
    Code_0(rangeDown);
    fprintf("****************\nEpoch: %d of %d\n***************\n", epoch, numEpochs);
    [summary, accuracy_by_episode] = qnn.train(verbose_level-1);
    
    
    recognition_accuracy(1,epoch) = (summary(1)*100)/(summary(1)+summary(4));
    classif_mode_ok_accuracy(1,epoch) = (summary(2)*100)/(summary(2)+summary(5));
    classif_by_window_accuracy(1,epoch) = (summary(3)*100)/(summary(3)+summary(6));
    
    fprintf("************************\nTRAINING Summary wins and losses\n************************\n");
    fprintf("Episodes won: %d, Episodes lost: %d\n", summary(1), summary(4)); 
    fprintf("Classif mode ok: %d, Classif mode NOT ok: %d\n", summary(2), summary(5)); 
    fprintf("Total wins window: %d, Total losses window: %d\n", summary(3), summary(6)); 
    fprintf("************************\nTRAINING Summary ACCURACY\n************************\n");
    fprintf("Recognition accuracy: %2.2f%%\n", recognition_accuracy(1,epoch)); 
    fprintf("Classif mode accuracy: %2.2f%%\n", classif_mode_ok_accuracy(1,epoch)); 
    fprintf("Wins in window accuracy: %2.2f%%\n", classif_by_window_accuracy(1,epoch)); 
    
    
    figure(1);
    plot(1:length(accuracy_by_episode),accuracy_by_episode, 'b');
    
    % plot(1:length(accuracy(1,:)),accuracy(1,:));
    mean_acuracy_by_epoch(1, epoch) = mean(accuracy_by_episode);
end
fprintf("Num. Trainings with samples: known=%d,dont known=%d\n", qnn.known, qnn.dont_known);
elapsedTimeHours = toc(tStart)/3600;
fprintf('\nElapsed time for training: %3.3f h\n\n', elapsedTimeHours);

training_accuracy = (mean(recognition_accuracy) + mean(classif_mode_ok_accuracy) + mean(classif_by_window_accuracy))/3;
model_dir = strcat("results/models/", s);
theta = qnn.theta;
save(model_dir,'theta');

tStart = tic;
Code_0(rangeDown+RepTraining);
[summary_test, accuracy_by_episode_test] = qnn.test(verbose_level-1, 13);
elapsedTimeHours = toc(tStart)/3600;
fprintf('Elapsed time for testing: %3.3f h \n', elapsedTimeHours);

figure(1);
hold on;
plot(1:length(accuracy_by_episode_test),accuracy_by_episode_test, 'r');
hold off;
xlabel('episode')
ylabel('Accuracy for windows predict')
legend('training', 'validation')
title('Accuracy for episode')
grid on;


test_recognition_accuracy = (summary_test(1)*100)/(summary_test(1)+summary_test(4));
test_classif_mode_ok_accuracy = (summary_test(2)*100)/(summary_test(2)+summary_test(5));
test_classif_by_window_accuracy = (summary_test(3)*100)/(summary_test(3)+summary_test(6));

fprintf("************************\nVALIDATION: Summary wins and losses\n************************\n");
fprintf("Episodes won: %d, Episodes lost: %d\n", summary_test(1), summary_test(4)); 
fprintf("Classif mode ok: %d, Classif mode NOT ok: %d\n", summary_test(2), summary_test(5)); 
fprintf("Total wins window: %d, Total losses window: %d\n", summary_test(3), summary_test(6)); 
fprintf("************************\nTEST: Summary ACCURACY\n************************\n");
fprintf("Recognition accuracy: %2.2f%%\n", test_recognition_accuracy(1,epoch)); 
fprintf("Classif mode accuracy: %2.2f%%\n", test_classif_mode_ok_accuracy(1,epoch)); 
fprintf("Wins in window accuracy: %2.2f%%\n", test_classif_by_window_accuracy(1,epoch)); 
    
    
test_accuracy = (test_recognition_accuracy + test_classif_mode_ok_accuracy + test_classif_by_window_accuracy)/3;


figure(2)
plot(1:length(qnn.training_cost), qnn.training_cost);
xlabel('update NN')
ylabel('Cost')
legend('training')
title('Cost by episode update')
end

