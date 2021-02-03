function [training_accuracy, test_accuracy, qnn] = QNN_emg_Exp_Replay(...
    params, emg_window_size, emg_stride, model_name, verbose_level, RepTraining, repTrainingForTesting)
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
RepTraining = 10;       % uo to 125 numero de muestras que voy a usar en el entrenamiento x cada usuario en la carpeta C:\Users\juanp\Desktop\QNN - EMG - RandomData - Copy\Data\Specific
repTrainingForTesting = 5;
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
    if verbose_level >= 1
        disp('Data conversion already done');
    end
else
    % Data conversion needed
    jsontomat;
end
    
% parameters for data/signals of users
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
transferFunctions = {'none', 'relu', 'relu', 'purelin'};


qnnOption = QNNOption(params.typeWorld, numNeuronsLayers, transferFunctions, ...
                params.lambda, params.learningRate, params.numEpochsToIncreaseMomentum, ...
                params.momentum, params.initialMomentum, ...
                params.miniBatchSize, params.W, params.gamma, params.epsilon);


qnn = QNN(qnnOption, params.rewardType, RepTraining, params.reserved_space_for_gesture);
qnn.initTheta(randInitializeWeights(qnn.qnnOption.numNeuronsLayers))
% con esta inicialización el entrenamiento no sirve ???
% theta2 = initWeights(qnn.qnnOption.numNeuronsLayers, -1, 1);



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
    if verbose_level >= 1
        fprintf("****************\nEpoch: %d of %d\n***************\n", epoch, numEpochs);
    end
    [summary_episodes, summary_classifications_mode, wins_by_episode, loses_by_episode] = qnn.train(verbose_level-1);
    
    
    recognition_accuracy(1,epoch) = (sum(summary_episodes(1, :))*100)/(sum(summary_episodes(1, :))+sum(summary_episodes(2, :)));
    classif_mode_ok_accuracy(1,epoch) = (sum(summary_classifications_mode(1, :))*100)/(sum(summary_classifications_mode(1, :))+sum(summary_classifications_mode(2, :)));
    classif_by_window_accuracy(1,epoch) = (sum(sum(wins_by_episode))*100)/(sum(sum(wins_by_episode))+sum(sum(loses_by_episode)));
    
%     fprintf("************************\nTRAINING Summary wins and losses\n************************\n");
%     fprintf("Episodes won: %d, Episodes lost: %d\n", sum(summary_episodes(1, :)), sum(summary_episodes(2, :))); 
%     fprintf("Classif mode ok: %d, Classif mode NOT ok: %d\n", sum(summary_classifications_mode(1, :)), sum(summary_classifications_mode(2, :))); 
%     fprintf("Total wins window: %d, Total losses window: %d\n", sum(sum(wins_by_episode)), sum(sum(loses_by_episode))); 
    
    
    if epoch == numEpochs && verbose_level >= 0
        
        figure(2);
        wins_t = wins_by_episode';
        losses_t = loses_by_episode';
        wins_in_total_episodes = wins_t(:);
        losses_in_total_episodes = losses_t(:);

        accuracy_by_episode = wins_in_total_episodes ./ (wins_in_total_episodes + losses_in_total_episodes);
        plot(1:length(accuracy_by_episode),accuracy_by_episode, 'b');
        % plot(1:length(accuracy(1,:)),accuracy(1,:));
        mean_acuracy_by_epoch(1, epoch) = mean(accuracy_by_episode);
    end
    
    
    
end
% fprintf("Num. Trainings with samples: known=%d,dont known=%d\n", qnn.known, qnn.dont_known);
elapsedTimeHoursTrain = toc(tStart)/3600;
training_accuracy = (mean(recognition_accuracy) + mean(classif_mode_ok_accuracy) + mean(classif_by_window_accuracy))/3;


%%% TESTING
rangeDown_test=rangeDown+RepTraining;  %limite inferior de rango de muestras a leer
assignin('base','rangeDown', rangeDown_test); 
assignin('base','RepTraining',  repTrainingForTesting); % ?? hace algo esto?

tStart = tic;
Code_0(rangeDown_test);
[test_summary_episodes, test_summary_classifications_mode, test_wins_by_episode, test_loses_by_episode] = qnn.test(verbose_level-1, repTrainingForTesting);
elapsedTimeHoursTest = toc(tStart)/3600;

test_recognition_accuracy = (sum(test_summary_episodes(1, :))*100)/(sum(test_summary_episodes(1, :))+sum(test_summary_episodes(2, :)));
test_classif_mode_ok_accuracy = (sum(test_summary_classifications_mode(1, :))*100)/(sum(test_summary_classifications_mode(1, :))+sum(test_summary_classifications_mode(2, :)));
test_classif_by_window_accuracy = (sum(sum(test_wins_by_episode))*100)/(sum(sum(test_wins_by_episode))+sum(sum(test_loses_by_episode)));

% fprintf("************************\nTESTING Summary wins and losses\n************************\n");
% fprintf("Episodes won: %d, Episodes lost: %d\n", sum(test_summary_episodes(1, :)), sum(test_summary_episodes(2, :))); 
% fprintf("Classif mode ok: %d, Classif mode NOT ok: %d\n", sum(test_summary_classifications_mode(1, :)), sum(test_summary_classifications_mode(2, :))); 
% fprintf("Total wins window: %d, Total losses window: %d\n", sum(sum(wins_by_episode)), sum(sum(test_loses_by_episode))); 

% fprintf("************************\nTRAINING Summary ACCURACY\n************************\n");
% fprintf("%2.2f,%2.2f,%2.2f\n", recognition_accuracy(1,epoch), classif_mode_ok_accuracy(1,epoch), classif_by_window_accuracy(1,epoch)); 
%     
% fprintf("************************\nTESTING Summary ACCURACY\n************************\n");
% fprintf("%2.2f %2.2f %2.2f\n", test_recognition_accuracy, test_classif_mode_ok_accuracy, test_classif_by_window_accuracy); 
test_accuracy = (test_recognition_accuracy + test_classif_mode_ok_accuracy + test_classif_by_window_accuracy)/3;
    
if verbose_level >= 1
    figure(2);
    wins_t = test_wins_by_episode';
    losses_t = test_loses_by_episode';
    test_wins_in_total_episodes = wins_t(:);
    test_losses_in_total_episodes = losses_t(:);

    accuracy_by_episode_test = test_wins_in_total_episodes ./ (test_wins_in_total_episodes + test_losses_in_total_episodes);

    hold on;
    plot(1:length(accuracy_by_episode_test),accuracy_by_episode_test, 'r');
    hold off;
    xlabel('episode')
    ylabel('Accuracy for windows predict')
    legend('training', 'validation')
    title('Accuracy for episode')
    grid on;



    figure(3)
    plot(1:length(qnn.cost), qnn.cost);
    xlabel('update NN')
    ylabel('Cost')
    legend('training')
    title('Cost by episode update')




    fprintf('\nElapsed time for training: %3.3f h\n', elapsedTimeHoursTrain);
    fprintf('Elapsed time for testing: %3.3f h \n', elapsedTimeHoursTest);


    fprintf("************************\nSummary Parameters, results\n************************\n");
    fprintf("%d %d %d %d %.2f %d %d %d %d %.2f\n", ...
        40, params.neurons_hidden1, params.neurons_hidden2, 6, params.learningRate, ...
        emg_window_size, emg_stride, params.miniBatchSize, params.reserved_space_for_gesture, ...
        params.epsilon);

    disp([recognition_accuracy(1,epoch), classif_mode_ok_accuracy(1,epoch), classif_by_window_accuracy(1,epoch), test_recognition_accuracy, test_classif_mode_ok_accuracy, test_classif_by_window_accuracy]);

end


end

