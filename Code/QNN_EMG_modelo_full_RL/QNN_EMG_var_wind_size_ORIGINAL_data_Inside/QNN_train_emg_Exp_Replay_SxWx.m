function QNN_train_emg_Exp_Replay_SxWx(USER_ID, experiment_begin, ...
    experiment_end, make_validation_too, write_excel, windows_sizes, strides, numRealEpochs)

%{
    Parameters:
        USER_ID: int -> número del usuario. Ejms: 1,3,5,306
        experiment_begin: int -> en el CSV de expériments, el identificador
            del experimento de inicio. Ejm: 27
        experiment_end: int -> en el CSV de expériments, el identificador
            del experimento de inicio. Ejm: 32
        make_validation_too: bool -> si es verdadero hace también
            validación usando rangedown=26+repTraining del entrenamiento,
            caso contrario la validación no se ejecuta.
        write_excel: bool -> si es verdadero escribirá resultados en
            experiments, caso contrario no escribirá esas filas
        windows_sizes: array -> vector con todos los tamaños de ventana a
            testear. Ejm: [200 250 300]
        strides: array -> vector con todos los strides para probar con cada
            uno de los tamaños de ventana anteriormente descritos. 
            Ejm: [20 40 50]
        numRealEpochs: int -> número de épocas. Es decir, número de veces
            que se le mostrará los mismos datos al modelo. Ejm: 10
    
  Ejemplo de llamada, si tengo en specific al usuario 288 y quiero probar
  del experimento 27 al 30 con validación escribiendo en excel. Para
  solamente el tamaño de ventana 200 y strides 100, 50. Se le muestra dos
  veces los mismos datos (real-epochs es 2)
  
  QNN_train_emg_Exp_Replay_SxWx(288, 27, 30, true, true, [200], [50 100], 2)    
%}
rng(1,'philox')
clc;
close all;
warning off all;

%%%%CAMBIODANNYINICIO%%%%
global user_id;
user_id = USER_ID;

if nargin < 4
    make_validation_too=true;
else
    if nargin < 5
       write_excel=true; 
    else
       if nargin < 6
           windows_sizes = [200]; 
       else
           if nargin < 7
               strides = [20];
           end
       end
    end
    
   
end

addpath(genpath('../experiments'));
init_parameters();
global parameters_training
global verbose % if true, the program will print ALL and use CPU for that
global index_numNeuronsLayers_input
global index_numNeuronsLayers_output
global index_RepTraining
global index_SamplingType  % 0=Random package, NO USADO
global index_learningRate
global index_numEpochsToIncreaseMomentum
global index_miniBatchSize
global index_lambda
global filename_experimentsQNN



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

row_position = 3;  % begins in A3 o excel to AN
%%%%PARAMETRIZACION_DANNY%%%%
for experiment_id=experiment_begin:experiment_end
    

    % Architecture of the artificial neural network
    %%%%PARAMETRIZACION_DANNY%%%%
    numNeuronsLayers = parameters_training(experiment_id, index_numNeuronsLayers_input:index_numNeuronsLayers_output);              %CAMBIAR [40, 75, 50, 6]; - #clases  [64, 75, 50, 6]
    transferFunctions{1} = 'none';
    transferFunctions{2} = 'relu';
    transferFunctions{3} = 'relu';
    transferFunctions{4} = 'purelin';
    options.reluThresh = 0;
    %%%%PARAMETRIZACION_DANNY%%%%
    options.lambda = parameters_training(experiment_id, index_lambda); % CAMBIAR regularization term
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
    for index_window=1:numel(windows_sizes)
        window_size = windows_sizes(index_window);
        %%%%PARAMETRIZACION_DANNY%%%%
        assignin('base','WindowsSize',  window_size);   %CAMBIAR - 200
        
        %%%%PARAMETRIZACION_DANNY%%%%
        for index_stride=1:numel(strides)
            stride = strides(index_stride);
            
            %%%%CAMBIODANNYINICIO%%%%
            % if stride==20 && window_size==200
                %estos experimentos ya se hicieron
            %     continue;
            %end
            fprintf("User id: %d, Experiment: %d, Window size: %d, stride:%d\n", USER_ID, experiment_id, window_size, stride);
            %%%%CAMBIODANNYFIN%%%%
            
            %%%%PARAMETRIZACION_DANNY%%%%
            assignin('base','Stride',  stride);        %CAMBIAR - 20

            %%%%PARAMETRIZACION_DANNY%%%%
            RepTraining        = parameters_training(experiment_id, index_RepTraining);       % uo to 125 numero de muestras que voy a usar en el entrenamiento x cada usuario en la carpeta C:\Users\juanp\Desktop\QNN - EMG - RandomData - Copy\Data\Specific
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
            
            % Random initialization of the weights
            theta = randInitializeWeights(numNeuronsLayers); 
			
			accuracy_by_epoch = zeros(1, numRealEpochs);
            
            for realEpoch=1:numRealEpochs
                
                general_file = "_windowSize"+window_size+"_stride"+stride+"_epoch"+realEpoch;
                s2 = ""+experiment_id+general_file;
                assignin('base','s2', s2);  
                
                if write_excel
                    dir_filename_window_stride = replace(filename_experimentsQNN, ".xlsx", general_file+".xlsx");
                    write_excel_training_wind_stride_base(dir_filename_window_stride, USER_ID, 1);
                    

                end
                %%%%CAMBIODANNYFIN%%%%
                
                fprintf("***********\nREAL EPOCH: %d of %d\n***********\n", realEpoch, numRealEpochs);

                rangeDown=26;  %limite inferior de rango de muestras a leer
                assignin('base','rangeDown', rangeDown); 
                Code_0(rangeDown);
                dataPacketSize1     = evalin('base', 'dataPacketSize');

                % Esto es nuevo 
                %Code_0;
                %dataPacketSize1   = evalin('base', 'dataPacketSize');


                %RepTraining      = 100;  % CAMBIAR numero de repeticiones por cada usuario (CAMBIAR SEGUN SE REQUIERA - up to 300)  CAMBIAR CAMBIAR CAMBIAR
                %assignin('base','RepTraining',RepTraining);
                options.numEpochs  = RepTraining*(dataPacketSize1-2);  %numero total de muestras de todos los usuarios

                % Momentum update
                %%%%PARAMETRIZACION_DANNY%%%%
                options.learningRate = parameters_training(experiment_id, index_learningRate);            % 9e-3  CAMBIAR
                options.typeUpdate = 'momentum';
                options.momentum = 0.9;
                options.initialMomentum = 0.3;
                %%%%PARAMETRIZACION_DANNY%%%%
                options.numEpochsToIncreaseMomentum = parameters_training(experiment_id, index_numEpochsToIncreaseMomentum); %1000

                
                %%%%PARAMETRIZACION_DANNY%%%%                   %CAMBIAR # de EXPERIMENTO
                options.miniBatchSize = parameters_training(experiment_id, index_miniBatchSize); %CAMBIAR % Size of the minibatch from experience replay

                % Window length for data smoothing
                options. W = 25;                   

                % Q-learning settings
                options.gamma = 1; % Q-learning parameter
                options.epsilon = 1.00; %Initial value of epsilon for the epsilon-greedy exploration
                % Fitting the ANN
                flagDisplayWorld = false;
                tStart = tic;


                %%%%PARAMETRIZACION_DANNY Y CAMBIO EN trainQNN (verbose) y summary%%%%
                [weights, summary, theta] = trainQNN_Exp_Replay(theta, numNeuronsLayers, transferFunctions, options, typeWorld, flagDisplayWorld, false);

                accuracy_by_epoch(1, realEpoch) = (summary(1)/(summary(1)+summary(4)) + summary(2)/(summary(2)+summary(5)) + summary(3)/(summary(3)+summary(6)))/3;
                elapsedTimeHours = toc(tStart)/3600;
                fprintf('\n Elapsed time for training: %3.3f h \n', elapsedTimeHours);
            
                
                s1 = 'QNN_Trained_Model_';
                s3 = '.mat';
                s = strcat(s1,s2,s3); 
                
                %validation
                save('QNN_Trained_Model.mat','weights', 'numNeuronsLayers',...
                'transferFunctions', 'options', 'typeWorld', 'flagDisplayWorld');

                %%%%CAMBIODANNYINICIO%%%%
                model_dir = strcat("results/models/", s);
                save(model_dir,'weights', 'numNeuronsLayers',...
                    'transferFunctions', 'options', 'typeWorld', 'flagDisplayWorld');

                %%%%CAMBIODANNYFIN%%%%


                %%%%CAMBIODANNYINICIO%%%%
                % put results in excel
                if write_excel
                    write_experiment_wind_stride_row(dir_filename_window_stride, parameters_training, row_position, summary, experiment_id, USER_ID);
                end

                %%%%CAMBIODANNYFIN%%%%
                
                
            
            end
            
            
            if numRealEpochs > 1
                a1 = 'FigEpochs_Exp_';
                a3 = '.png';
                s2 = ""+experiment_id+"_windowSize"+window_size+"_stride"+stride+"_numEpochs"+numRealEpochs;
                
                saux = strcat("results/figures/", a1,s2,a3);
                figure(5);
                plot(1:numel(accuracy_by_epoch), accuracy_by_epoch);
                saveas(gcf,saux)
                close all;
            end
            
            
        end
    end
    row_position = row_position + 1;

end

%%%%CAMBIODANNYINICIO%%%%
row_position = row_position + 1;


if make_validation_too
    QNN_validation_Exp_Replay_SxWx(USER_ID, row_position, ...
        26+parameters_training(experiment_id, index_RepTraining), ...
        experiment_begin, experiment_end, write_excel, windows_sizes, strides, numRealEpochs);
end
%%%%CAMBIODANNYFIN%%%%
end

