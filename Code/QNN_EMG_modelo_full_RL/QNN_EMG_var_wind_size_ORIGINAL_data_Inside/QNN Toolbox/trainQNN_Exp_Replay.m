function [weights, summary, theta] = trainQNN_Exp_Replay(th, ...
    numNeuronsLayers, transferFunctions, options, typeWorld, ...
    flagDisplayWorld, verbose, use_memory_action)
%%%%CAMBIODANNYINICIO%%%%
% default parameter
theta = th;
summary = [];
if nargin<6
  verbose = true;
end
%%%%CAMBIODANNYFIN%%%%

% Window length for data smoothing
W = options. W;                                                         % Window length for data smoothing
% Initial settings
learningRate = options.learningRate;                                    % 9e-3; learningRate
numEpochs = options.numEpochs;                                          % options.numEpochs = 100000;
                       % randInitializeWeights([64, 75, 50, 4]);
% Learning rate for training the ANN
alpha = options.learningRate;                                           % 9e-3; -> learningRate = alpha
% Epsilon
epsilon0 = options.epsilon;                                             %Initial value of epsilon for the epsilon-greedy exploration = 1
epsilon = epsilon0;
if strcmp(options.typeUpdate, 'momentum')                                %  Momentum update options.typeUpdate='momentum' (string)
    % Setup for momentum update of the weights of the ANN
    momentum = options.initialMomentum;                                  %  InitialMomentum -> options.initialMomentum = 0.3;
    velocity = zeros( size(theta) );                                     %  Creo vector de ceros 'velocity' del tamaño de los pesos theta
    numEpochsToIncreaseMomentum = options.numEpochsToIncreaseMomentum;   %  options.numEpochsToIncreaseMomentum = 1000;
else
    error('Invalid selection for updating the weights \n');
end
% Gamma
gamma = options.gamma;                                                   % options.gamma = 1; % Q-learning parameter
% Training the ANN
cost = zeros(1, numEpochs);                                              % vector de zeros 'cost' de dimension  options.numEpochs = 100000;
% Initializing the history of the average reward
averageEpochReward = zeros(1, numEpochs);                                % vector de zeros 'averageEpochReward' de dimension  options.numEpochs = 100000;
cumulativeEpochReward = 0;                                               % inicializo cumulativeEpochReward = 0;

%****************************************************************
% Intializing the history of the average of the maxQ values
%numero de valores de Q depende de tipo de mundo en el juego
%%%% Verificar info: deterministic, randAgent, and randWord ????
% if strcmp(typeWorld, 'deterministic') || strcmp(typeWorld, 'randAgent')  % Type of the world of the game: deterministic, randAgent, and randWord
%     tw = 'randAgent';
%     numStates = nchoosek(16, 4)*factorial(4);                            % 16 ubicaciones, 4 posibles acciones?  (CAMBIAR - REVISAR)
% elseif strcmp(typeWorld, 'randWorld')
%     tw = 'randWorld';
%     numStates = 13;                                                      % 13 posibles estados (-1 start, -1 win state, -1 lose state)
% end                                                                      % (CAMBIAR - REVISAR)
%*****************************************************************

%$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
% if strcmp(typeWorld, 'deterministic') || strcmp(typeWorld, 'randAgent')  % Type of the world of the game: deterministic, randAgent, and randWord
%     tw = 'randAgent';
%     numStates = nchoosek(16, 6)*factorial(6);                            % 16 ubicaciones, 4 posibles acciones?  (CAMBIAR - REVISAR)
% elseif strcmp(typeWorld, 'randWorld')
%     tw = 'randWorld';
%     numStates = 13;                                                      % 13 posibles estados (-1 start, -1 win state, -1 lose state)
% end
%$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
state_length = 40;
if use_memory_action
    state_length = 46;
end


numTestingSamples = max(0.1*state_length, 20);                                       % numTestingSamples = max(0.1*numStates, 20);   ????????
testStates = zeros(numTestingSamples, state_length);                                 % ???????????????????????????????????????

% for i = 1:numTestingSamples                                              % ??????????
%     stateAux = rand(1,40); %createWorld(tw);
%     testStates(i, :) = stateAux(:)';
% end

averageMaxQvalue = zeros(1, numEpochs);                                  %Inicializo 'averageMaxQvalue' vector de zeros (1 val x epoch)
% Maximum number of interactiobs allowed********************
%maxIterationsAllowed = 100;                                              %%
%********************************************************

%maxWindowsAllowed = 40;  %# de ventanas maximo x muestra                    %AQUI PONER NUMERO # DE VENTANAS EMG  rev
%EMG_window_size = 200;                                                  %AQUI PONER WINDOW SIZE
%stride = 20;                                                            %AQUI PONER STRIDE

EMG_window_size = evalin('base', 'WindowsSize');                                                %AQUI PONER WINDOW SIZE
Stride = evalin('base', 'Stride'); 

% History of wins
% %Inicializo 'countWins = 0' & 'historyAverageWins (numEpochs)'
%countWins = 0;
historyAverageWins = zeros(1, numEpochs);


% Experience replay initialization ------------------------------
miniBatchSize = options.miniBatchSize;
lengthBuffer = ceil(3*miniBatchSize);



s  =  nan(lengthBuffer, state_length);
sp =  nan(lengthBuffer, state_length);
a  =  nan(lengthBuffer, 1);
r  =  nan(lengthBuffer, 1);
gameReplay = [s, a, r, sp]; %(state, action, reward, state')
h = 0;
numIterationTotal = 0;

%---------------------------------------------------------------


%-----------------vvvvv for epoch = 1:numEpochs vvvvvvv------------------------------
%-----------------------------------------------------------------------
%-----------------------------------------------------------------------
%-----------------------------------------------------------------------

%2222222222222222222222222222222222222222222
rangeDown   = evalin('base', 'rangeDown');
Code_0(rangeDown);
                                      

dataPacketSize   = evalin('base', 'dataPacketSize');
orientation      = evalin('base', 'orientation');
RepTraining      =  evalin('base', 'RepTraining'); %RepTraining1; 2 %;  %options.epsilon +2; %2;  %numero de repeticiones por cada usuario (CAMBIAR SEGUN SE REQUIERA - up to 300)  CAMBIAR CAMBIAR CAMBIAR
RepTotalTraining =  RepTraining*(dataPacketSize-2);
Reward_type      =  evalin('base', 'Reward_type');
post_processing      =  evalin('base', 'post_processing');

window_n=0; %contador de numero de ventanas
conta=1;

%%%% DANNY NUM WINDOWS
dataPacket = evalin('base','dataPacket');
userIndex   = evalin('base', 'userIndex');
userData = loadSpecificUser(dataPacket, userIndex+2);
energy_index = strcmp(orientation(:,1), userData.userInfo.name);
rand_data=orientation{energy_index,6};
emgRepetition = evalin('base','emgRepetition');
numberPointsEmgData = length(userData.training{rand_data(emgRepetition),1}.emg);
% -1 to be in alignment with Victor's calculus
num_windows = getNumberWindows(numberPointsEmgData, EMG_window_size, Stride, false)-1;
       
etiquetas_labels_predichas_matrix=strings(num_windows,numEpochs);

countWins=0;
countLoses=0;

countWins2=0;
countLoses2=0;

number_classif_ok=0;
number_classif_failed=0;

%%%%CAMBIODANNYINICIO%%%%
debug_step = ceil(numEpochs/5);
%%%%CAMBIODANNYFIN%%%%

for epoch = 1:numEpochs %numero total de muestras de todos los usuarios
    % Creating a new instance of the world
    
    %%%%CAMBIODANNYINICIO%%%%
    if mod(epoch, debug_step) == 0
        fprintf("Epoch %d of %d\n", epoch, numEpochs);
    end
    %%%%CAMBIODANNYFIN%%%%
    
    cumulativeIterationReward = 0;
    numIterationInCurrentEpisode = 0;
    
    %*********************************
    %state = createWorld(typeWorld);                     %obtengo estado inicial S desde el ambiente  (AQUI HAY QUE CAMBIAR - features de ventana EMG)
    %dim state vector = (4,4,4) % 4 matrices de 4x4
    %**********************************
    %disp('epoch');disp(epoch);
    
    %Leo datos de muestras --------------------------------------------
    %%%%PARAMETRIZACION_DANNY%%%%
    [Numero_Ventanas_GT,EMG_GT,Features_GT,Tiempos_GT,Puntos_GT,Usuario_GT,gestureName_GT,groundTruthIndex_GT,groundTruth_GT] = ...
        Code_1(orientation,dataPacketSize,RepTraining,RepTotalTraining, verbose);
    
    while Usuario_GT=="NaN"
        
        if Usuario_GT~="NaN"
            break
        end
        %%%%PARAMETRIZACION_DANNY%%%%
        [Numero_Ventanas_GT,EMG_GT,Features_GT,Tiempos_GT,Puntos_GT,Usuario_GT,gestureName_GT,groundTruthIndex_GT,groundTruth_GT] = ...
            Code_1(orientation,dataPacketSize,RepTraining,RepTotalTraining, verbose);
        
    end
    %disp("Numero_Ventanas_GT"); disp(Numero_Ventanas_GT);
    %         disp("EMG_GT"); disp(EMG_GT);
    %         disp("gestureName_GT"); disp(gestureName_GT);
    %disp("groundTruthIndex_GT"); disp(groundTruthIndex_GT);
    %         disp("groundTruth_GT"); disp(groundTruth_GT);
    %         disp("EMG_GT"); disp(EMG_GT);
    
    %Creo vector con limite derecho de cada ventana
    gt_gestures_pts=zeros(1,Numero_Ventanas_GT);
    gt_gestures_pts(1,1)=EMG_window_size;
    
    for k = 1:Numero_Ventanas_GT-1
        gt_gestures_pts(1,k+1)=gt_gestures_pts(1,k)+Stride;
    end
    %disp('gt_gestures_pts');disp(gt_gestures_pts);
    assignin('base','gt_gestures_pts',gt_gestures_pts);
    
    gt_gestures_labels = mapGroundTruthToLabelsWithPts(gt_gestures_pts, groundTruthIndex_GT, gestureName_GT, 0.2);
%     %Creo vector de etiquetas para Ground truth x ventana
%     gt_gestures_labels=strings;
%     %gt_gestures_labels(1,1)="noGesture";
%     for k2 = 1:Numero_Ventanas_GT
%         if gt_gestures_pts(1,k2) > (groundTruthIndex_GT(1,1) + EMG_window_size/5) && gt_gestures_pts(1,k2) < groundTruthIndex_GT(1,2)
%             %disp('case1')
%             gt_gestures_labels(1,k2)=string(gestureName_GT);
%         elseif  gt_gestures_pts(1,k2)-EMG_window_size < (groundTruthIndex_GT(1,2)-EMG_window_size/5 ) && gt_gestures_pts(1,k2) > groundTruthIndex_GT(1,2)
%             %disp('case2')
%             gt_gestures_labels(1,k2)=string(gestureName_GT);
%         else
%             %disp('case3')
%             gt_gestures_labels(1,k2)="noGesture";
%         end
%     end
    %disp('gt_gestures_labels');disp(gt_gestures_labels);
    assignin('base','gt_gestures_labels',gt_gestures_labels);
    %Creo vector de etiquetas de Ground truth con valores numericos
    gt_gestures_labels_num=zeros(Numero_Ventanas_GT-1,1);
    %gt_gestures_labels_num(1,1)=6;
    for k3 = 2:Numero_Ventanas_GT
        if gt_gestures_labels(1,k3) == "waveOut"
            gt_gestures_labels_num(k3-1,1)=1;
        elseif gt_gestures_labels(1,k3) == "waveIn"
            gt_gestures_labels_num(k3-1,1)=2;        %CAMBIAR
        elseif gt_gestures_labels(1,k3) == "fist"
            gt_gestures_labels_num(k3-1,1)=3 ;         %CAMBIAR
        elseif gt_gestures_labels(1,k3) == "open"
            gt_gestures_labels_num(k3-1,1)=4;         %CAMBIAR
        elseif gt_gestures_labels(1,k3) == "pinch"
            gt_gestures_labels_num(k3-1,1)=5;       %CAMBIAR
        elseif gt_gestures_labels(1,k3) == "noGesture"
            gt_gestures_labels_num(k3-1,1)=6;        %CAMBIAR
        end
    end
    %disp('gt_gestures_labels_num');disp(gt_gestures_labels_num);
    assignin('base','gt_gestures_labels_num',gt_gestures_labels_num);
    %----------------------------------------------
    
    
    %---- Obtengo datos de primera ventana de cada muestra e inicializo vectores de ground truth-----------------*
    window_n=window_n+1;                               %window_n == 1
    Vector_EMG_Tiempos_GT=zeros(1,Numero_Ventanas_GT); %creo vector de tiempos de gt
    Vector_EMG_Puntos_GT=zeros(1,Numero_Ventanas_GT);  %creo vector de puntos de gt
    Vector_EMG_Tiempos_GT(1,window_n)=Tiempos_GT;      %copio primer valor en vector de tiempos de gt
    Vector_EMG_Puntos_GT(1,window_n)=Puntos_GT;        %copio primer valor en vector de tiempos de gt
    
    %---- Defino estado inicial en base a cada ventana EMG  -----------------------------------------------------*
    %state = rand(1, 40);
    
    
    state = table2array(Features_GT);           %Defino ESTADO inicial
    
    
    state = (state - min(state)) / ( max(state) - min(state) );
    if use_memory_action
        mask_state = [0 0 0 0 0 1];
        state = [state, mask_state];
    end
    
    
    %%%%CAMBIODANNYINICIO%%%%
    if verbose
        disp('initial state')
    end
    %%%%CAMBIODANNYFIN%%%%
    
    %disp(state)
    %disp('state')
    %disp(size(state))
    
    %Each_complete_signal = rand(1, 40*Numero_Ventanas_GT); %rand(1, 40*maxWindowsAllowed);      %AQUI CAMBIAR, HAY QUE PONER CADA SEñAL COMPLETA AQUI  %&&&&&&&&&&&&&&&&
    
    % ---- Inicializo variables requeridas para guardar datos de prediccion
    % - Quito primera etiqueta de ventana ya que esa no se predice en el
    % lazo while
    etiquetas = gt_gestures_labels_num; %1+round((5)*rand(Numero_Ventanas_GT,1));   %1+round((5)*rand(maxWindowsAllowed,1)); %%%%%%%% AQUI PONER ground truth de cada ventana EMG - gestos de 1 a 6
    etiquetas_labels_predichas_vector=strings;
    %etiquetas_labels_predichas_matrix=strings;
    etiquetas_labels_predichas_vector_without_NoGesture=strings;
    %etiquetas_labels_predichas_matrix_without_NoGesture=strings;
    acciones_predichas_vector = zeros(Numero_Ventanas_GT,1);%zeros(Numero_Ventanas_GT,1) zeros(maxWindowsAllowed,1);         %%%%%%%%   EN ESTE VECTOR VAN A IR LAS ACCIONES PREDICHAS, LAS
    
    % ---- inicializo parametros medicion tiempo, y vectores de prediccion
    % necesarios para evaluar reconocimiento en cada epoca -----------------
    ProcessingTimes_vector=[];
    TimePoints_vector=[];
    n1=0;
    etiquetas_labels_predichas_vector_simplif=strings;
    
    %$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    %----------------------------------------
    % Showing the initial state of the world;
    %if flagDisplayWorld == true
    %    figure(1);
    %    displayWorld(state);
    %end
    %----------------------------
    
    % Interaction of agent with the world
    gameOn = true;                                               % Indicator of reaching a final state
    cumulativeGameReward = 0;                                    % Inicializo reward acumulado
    numIteration = 0;                                            % Inicializo  numIteration
    
    %     dataX = zeros(Numero_Ventanas_GT, 40); %zeros(maxWindowsAllowed, 40);                     % Inicializo vector de entrada de ANN (AQUI HAY QUE CAMBIAR) 64 neuronas entrada
    %
    %     %$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    %     dataY = zeros(Numero_Ventanas_GT, 6); %zeros(maxWindowsAllowed, 6);                      % Inicializo vector de salida de ANN (AQUI HAY QUE CAMBIAR)  4 neuronas salida
    %     %$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    
    %-----------------------------------------------------------------------
    %------------------%INICIO del episodio------------------------------------
    
    tic;   %comienza el conteo del tiempo de INICIO
    
    % -----  El juego viene a ser el barrido de ventanas en una muestra EMG
    % ----- Se gana el juego si el resultado de reconocimiento es correctom
    % ----- sino se pierde el juego y se tiene una penalizacion
    while gameOn %...............................................
        
        %-------------------------Exp Repa new ----------------------
        numIterationTotal = numIterationTotal + 1;
        numIterationInCurrentEpisode = numIterationInCurrentEpisode + 1;
        %------------------------------------------------------------
        
        numIteration = numIteration + 1;                          % incremento el numero de iteracion
        
        % Reshaping the weights of the ANN
        weights = reshapeWeights(theta, numNeuronsLayers);        % Reshaping the weights of the ANN (verificar shape)
        % Predicting the response of the ANN for the current state
        [dummyVar, A] = forwardPropagation(state, weights,...     % ANN to obtain update weights - state is a vector
            transferFunctions, options);
        
        Qval = A{end}(:, 2:end);                                  % 6 Valores de Q - uno por cada accion
        
        [dummyVar, idx] = max(Qval);                              % obtengo indice de Qmax a partir de vector Qval
        %disp('idxMax');disp(idx);
        % Epsilon-greedy action selection - REV Con epsilon=1
        % Inicialmente hace solo exploracion, luego el valor de epsilon se va reduciendo a medida que tengo mas informacion
        % Si rand <= epsilon, obtengo un Q de manera aleatoria, el cual será diferente a Qmax (exploracion)
        %disp('epsilon');disp(epsilon);
        if rand <= epsilon            % siempre se cumple con epsilon=1% %Initial value of epsilon for the epsilon-greedy exploration
            full_action_list = 1:6;    %actionList = 1:6;               % posibles acciones  (AQUI HAY QUE CAMBIAR - #clases) usar gestos
            actionList = full_action_list(full_action_list ~= idx);           %  Crea lista con las acciones q no tienen Qmax
            idx_valid_action = randi([1 length(actionList)]);
            idx = full_action_list(actionList(idx_valid_action));
            %             actionList = 1:6;    %actionList = 1:6;               % posibles acciones  (AQUI HAY QUE CAMBIAR - #clases) usar gestos
%             actionList = actionList(actionList ~= idx);           %  Crea lista con las acciones q no tienen Qmax
%             [dummyVar, idx] = sort( rand(1, 6) ); %rand(1, 3) rand(1, 5) %  creo vector randomico de 5 elementos [a, b, c, d, e] con valores de 1 a 5
%             %disp('idxRand');disp(idx(1));
        end
        
        % - Predigo accion en base a Epsilon-Greedy
        action = idx(1);                                    % elijo primer valor de idx
        acciones_predichas_vector(numIteration,1)=action;   % AQUI SE VAN GUARDADNO LAS ACCIONES PREDICHAS DENTRO DEl vector de UNA EPOCA
        
        
        
        %2222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222
        % Numero_Ventanas     %   Escalar: ejem: 40 ventanas
        % EMG                 %   200 points x 8 channels
        % Vector_EMG_Features %   1x40 feature column - table type
        % Vector_EMG_Tiempos  %   Escalar - es necesario vectorizar y procesar
        % Vector_EMG_Puntos   %   Escalar - es necesario vectorizar y procesar
        % Nombre_Usuario      %   String - ej: 'user3'
        % gestureName         %   categorical - ej: noGesture
        % groundTruthIndex    %   [0 1] since is Nogesture
        % groundTruth         %   1x1000 points logical array
        
        % WO    = 1 % WI    = 2 % FIST  = 3 % OPEN  = 4 % PINCH = 5 % RELAX = 6
        %              _________
        % 1 |---------|        |------------| 1000
        
        %%%%PARAMETRIZACION_DANNY%%%%
        [Numero_Ventanas_GT,EMG_GT,Features_GT,Tiempos_GT,Puntos_GT,Usuario_GT,gestureName_GT,groundTruthIndex_GT,groundTruth_GT] = ...
            Code_1(orientation,dataPacketSize,RepTraining,RepTotalTraining, verbose);
        
        while Usuario_GT=="NaN"
            if Usuario_GT~="NaN"
                break
            end
            %%%%PARAMETRIZACION_DANNY%%%%
            [Numero_Ventanas_GT,EMG_GT,Features_GT,Tiempos_GT,Puntos_GT,Usuario_GT,gestureName_GT,groundTruthIndex_GT,groundTruth_GT] = ...
                Code_1(orientation,dataPacketSize,RepTraining,RepTotalTraining, verbose);
        end
        
        window_n=window_n+1; %window_n == 2 hasta window final
        
        %---- Vectorizo datos necesarios de vectores de timpos y de puntos de Ground truth -----------------*
        %  if window_n==Numero_Ventanas_GT -> FIN DE JUEGO
        if window_n==Numero_Ventanas_GT%+1  %REV este +1  %window_n==Numero_Ventanas_GT
            Vector_EMG_Tiempos_GT(1,window_n)=Tiempos_GT;  % al final tengo vector de tiempos ej: (1xNumero_Ventanas_GT) - ultima ventana
            Vector_EMG_Puntos_GT(1,window_n)=Puntos_GT;    % al final tengo vector de puntos ej: (1xNumero_Ventanas_GT) - ultima ventana
            fin_del_juego=1;                               % bandera fin de juego
            window_n=0;                                    % reinicio cont ventana
            %%%%CAMBIODANNYINICIO%%%%
            if verbose
                disp('fin del juego')
            end
            %%%%CAMBIODANNYFIN%%%%
            
        else % EN ESTE ELSE se hace el barrido de ventanas, y se concatena datos en vectores
            Vector_EMG_Tiempos_GT(1,window_n)=Tiempos_GT;  %dato escalar guardo en vector
            Vector_EMG_Puntos_GT(1,window_n)=Puntos_GT;    %dato escalar guardo en vector
            fin_del_juego=0;
        end
        
        %---- Aqui pongo los nuevos estados, los cuales corresponden a las features de cada ventana de señal EMG -----
        new_state = table2array(Features_GT);
        new_state = (new_state - min(new_state)) / ( max(new_state) - min(new_state) );
        if use_memory_action
            mask_state = sparse_one_hot_encoding(action, 6);
            new_state = [new_state, mask_state];
        end
        %--------------------------------------------------------------
        
        %$$$$  REWARD - comparo if acciones_predichas_actual==etiqueta_actual - RECOMPENSA 0 siempre      -------
        %ahora no tengo este dato, x lo que le he puesto ranfomico, igual recompensa siempre es cero
        
        %disp("etiquetas");disp(etiquetas);
        %          disp(size(etiquetas))
        assignin('base','etiquetas',etiquetas);
        %disp("numIteration");disp(numIteration);
        assignin('base','numIteration',numIteration);
        etiqueta_actual=etiquetas(numIteration,1);                           % ground truth de cada ventana EMG - CAMBIAR -
        %disp("etiqueta_actual");disp(etiqueta_actual);
        assignin('base','etiqueta_actual',etiqueta_actual);
        
        acciones_predichas_actual=acciones_predichas_vector(numIteration,1); % accion predicha por ANN
        %disp("acciones_predichas_actual");disp(acciones_predichas_actual);
        assignin('base','acciones_predichas_actual',acciones_predichas_actual);
        
        
        if Reward_type ==true
            %disp('-1 reward')
            reward = getReward_emg(acciones_predichas_actual, etiqueta_actual); % AQUI HAY QUE DEFINIR RECOMPENSAS EN BASE A GROUNDTRUTH EMG
        else
            %disp('0 reward')
            reward = getReward_emg_0reward(acciones_predichas_actual, etiqueta_actual);
        end
        %%%%CAMBIODANNYINICIO%%%%
        if verbose && reward~=0
            disp("reward");
            disp(reward);
        end
        %%%%CAMBIODANNYFIN%%%%
        
       if  acciones_predichas_actual ~= etiqueta_actual  %conteo de clasificaciones exitosas
           countLoses2 = countLoses2 +1;
       else
           countWins2 = countWins2 + 1;
       end
        
        %reward = getReward_emg(acciones_predichas_actual, etiqueta_actual); % AQUI HAY QUE DEFINIR RECOMPENSAS EN BASE A GROUNDTRUTH EMG
        %getReward_emg_0reward  %- si se desea REWARD = 0
        %$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        
        
        %---- Experience replay storage ------------------------------------
        h = h + 1;
        idx = mod(h, lengthBuffer);
        if idx == 0
            idx = lengthBuffer;
        end
        
        
        gameReplay(idx, :) = [state, action, reward, new_state];   %[state(:)', action, reward, new_state(:)'];
        %---------------------------------------------------------------
        
        
        assignin('base','acciones_predichas_vector',acciones_predichas_vector);
        % asigno a gesto --1 a 6-- una etiqueta categorica  %ESTO CAMBIAR - Etiquetas de PREDICCIONES
        % class(etiquetas_labels_predichas_vector)
        if acciones_predichas_actual == 1
            etiquetas_labels_predichas_vector(numIteration,1)="waveOut";
        elseif acciones_predichas_actual == 2
            etiquetas_labels_predichas_vector(numIteration,1)="waveIn";        %CAMBIAR
        elseif acciones_predichas_actual == 3
            etiquetas_labels_predichas_vector(numIteration,1)="fist" ;         %CAMBIAR
        elseif acciones_predichas_actual == 4
            etiquetas_labels_predichas_vector(numIteration,1)="open";         %CAMBIAR
        elseif acciones_predichas_actual == 5
            etiquetas_labels_predichas_vector(numIteration,1)="pinch" ;       %CAMBIAR
        elseif acciones_predichas_actual == 6
            etiquetas_labels_predichas_vector(numIteration,1)="noGesture" ;        %CAMBIAR
        end
        
        %         disp('numIteration');disp(numIteration);
        %         disp('Numero_Ventanas_GT-1');disp(Numero_Ventanas_GT-1);
        
        %Acondicionar vectores - si el signo anterior no es igual al signo acual entocnes mido tiempo
        if numIteration>1 && numIteration~=Numero_Ventanas_GT-1 && ...   %numIteration~=maxWindowsAllowed
                etiquetas_labels_predichas_vector(numIteration,1) ~= etiquetas_labels_predichas_vector(numIteration-1,1)
            
            n1=n1+1;
            ProcessingTimes_vector(1,n1) = toc;  %mido tiempo transcurrido desde ultimo cambio de gesto
            tic;
            
            %obtengo solo etiqueta que se ha venido repetiendo hasta instante numIteration-1
            etiquetas_labels_predichas_vector_simplif(1,n1)=etiquetas_labels_predichas_vector(numIteration-1,1);
            
            %obtengo nuevo dato para vector de tiempos
            TimePoints_vector(1,n1)=Stride*numIteration+EMG_window_size/2;           %necesito dato de stride y tamaño de ventana de Victor
            
        elseif numIteration== Numero_Ventanas_GT-1 %==maxWindowsAllowed    % si proceso la ultima ventana de la muestra de señal EMG
            
            %disp('final window')
            
            n1=n1+1;
            ProcessingTimes_vector(1,n1) = toc;  %mido tiempo transcurrido desde ultimo cambio de gesto
            tic;
            
            %obtengo solo etiqueta que no se ha repetido hasta instante numIteration-1
            etiquetas_labels_predichas_vector_simplif(1,n1)=etiquetas_labels_predichas_vector(numIteration,1);
            
            %obtengo dato final para vector de tiempos
            kj=size(groundTruth_GT);  %  se supone q son 1000 puntos
            TimePoints_vector(1,n1)=  kj(1,2);                 %AQUI CAMBIAR  %1000 puntos
            
        end
        
        %Saco la moda de los gestos diferentes a NoGesture
        temp1=(size(etiquetas_labels_predichas_vector));
        temp1=temp1(1,1);
        
        %disp('etiquetas_labels_predichas_vector')
        %disp(etiquetas_labels_predichas_vector)
        %saco no gesture de este vector para poder usar funcion moda
        %-------- REVISAR-ESTO HABILITAR SI REQUIERO QUITAR NO GESTURE -----
        for i=1:temp1
            if etiquetas_labels_predichas_vector(i,1)~="noGesture"
                etiquetas_labels_predichas_vector_without_NoGesture(i,1)=etiquetas_labels_predichas_vector(i,1);
            else
            end
        end
        %disp('etiquetas_labels_predichas_vector_without_NoGesture')
        %disp(etiquetas_labels_predichas_vector_without_NoGesture)
        
        %dependiendo variable "noGestureDetection" Saco moda 1) sin considerar no gesto ,o 2) considerando no gesto
        noGestureDetection   = evalin('base', 'noGestureDetection');
        if noGestureDetection == false
            class_result=mode(categorical(etiquetas_labels_predichas_vector_without_NoGesture));  %Saco la moda de las etiquetas dif a no gesture
        elseif noGestureDetection == true
            class_result=mode(categorical(etiquetas_labels_predichas_vector));    %Saco moda incluyendo etiqueta de NoGesture
        end
        
        %Si al sacar la moda todo es no gesto (<missing>), entonces la moda es no gesto
        if ismissing(class_result)
            class_result="noGesture";
        else
        end
        
       %esto es para guardar resultado de clasificacion 
       class_result_vector(1,conta)=string(class_result); %
       %disp('Predicted Class');disp(class_result)
       assignin('base','class_result_vector',class_result_vector);
        

       

        
        %-----------------------------------------------------------------
        
        %class_result=mode(etiquetas_labels_predichas_vector);
        
        
        % Cumulative reward so far for the current episode
        %cumulativeGameReward = cumulativeGameReward + reward;        % recompensa acumulada
        %-----------------------------
        
        %****************************************************************
        %         % Q-Learning Algorithm
        %         % Getting the value of Q(s, a)                            %from state(:)' Getting the value of Q(s, a)
        %         [dummyVar, A] = forwardPropagation(state(:)', weights,... % SACA ESTO OTRA VEZ??  PARA SACAR OLD Q VALUE?????????????????
        %             transferFunctions, options);
        %         old_Qval = A{end}(:, 2:end);                              %OLD Q VALUE
        %         % Getting the value of max_a'{Q(s', a')}                  %from new_state(:)' Getting the value of max_a'{Q(s', a')}
        %         [dummyVar, A] = forwardPropagation(new_state(:)', weights,...
        %             transferFunctions, options);
        %         new_Qval = A{end}(:, 2:end);                              %NEW Q VALUE
        %         maxQval = max(new_Qval);
        %*****************************************************************
        
        % Q-Learning Algorithm
        % Getting the value of Q(s, a)                            %from state(:)' Getting the value of Q(s, a)
        %         [dummyVar, A] = forwardPropagation(state, weights,... % SACA ESTO OTRA VEZ??  PARA SACAR OLD Q VALUE?????????????????
        %             transferFunctions, options);
        %         old_Qval = A{end}(:, 2:end);
        %         %disp('old_Qval')
        %         %disp(old_Qval)
        %         %OLD Q VALUE
        %         % Getting the value of max_a'{Q(s', a')}                  %from new_state(:)' Getting the value of max_a'{Q(s', a')}
        %         [dummyVar, A] = forwardPropagation(new_state, weights,...
        %             transferFunctions, options);
        %         new_Qval = A{end}(:, 2:end);                              %NEW Q VALUE
        %         maxQval = max(new_Qval);
        
        %%%%%%%%%%%%%%%%%%%%%%%% Configurar cuanto estaré en un estado
        %%%%%%%%%%%%%%%%%%%%%%%% terminal (depende de numero de ventanas)
        
        %-----------------------------------------------------------------
        % Computation of the target
        %         if abs(reward) ~= 10
        %             % Target for a non-terminal state  (or learned value)
        %             %Ecuacion de funcion de valor Q(s)=R(t+1)+gamma*maxQvalue'
        %             target = reward + gamma*maxQval;
        %         else
        %             % Taget for a terminal state
        %             target = reward;
        %         end
        %------------------------------------------------------------------
        
        % %$$$$$$$$$$ AQUI DEFINO LA RECOMPENSA EN FUNCION DE SI ES EL ESTADO FINAL $$$$$$$$$$$$$
        %         if  numIteration ~= maxIterationsAllowed  % abs(reward) ~= 10
        %             % Target for a non-terminal state  (or learned value)
        %             %Ecuacion de funcion de valor Q(s)=R(t+1)+gamma*maxQvalue'
        %             target = reward + gamma*maxQval;    %valor de tarjet si aun no termina episodio
        %         else
        %             % Taget for a terminal state
        %             target = reward;                    %valor de tarjet para fin de episodio
        %         end
        % %$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        
        %**************************************************************
        %         if reward == -10 % end the game - lose
        %             resultGame = 'lost';
        %             gameOn = false;
        %         elseif reward == +10  % end the game - win
        %             resultGame = 'won ';
        %             countWins = countWins + 1;
        %             gameOn = false;
        %         end
        %***************************************************************
        
        
        
        %$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        %AQUI EVALUO EL RECONOCIMIENTO UNA VEZ QUE SE TERMINA EL EPISODIO
        %%%%CAMBIODANNYINICIO%%%%
        if verbose
            disp("Numero_Ventanas_GT");disp(Numero_Ventanas_GT-1)
        end
        %%%%CAMBIODANNYFIN%%%%
        assignin('base','Numero_Ventanas_GT',Numero_Ventanas_GT-1);
        if  numIteration == Numero_Ventanas_GT -1 %==maxWindowsAllowed % reward == -10  end the game - lose
            
            %-----------check de las variables predichas que entran a eval de reconocimiento
            %disp('etiquetas_labels_predichas_vector');
            %disp(etiquetas_labels_predichas_vector); %[N,1] %vector Full
            assignin('base','etiquetas_labels_predichas_vector',etiquetas_labels_predichas_vector);
            var1=size(etiquetas_labels_predichas_vector);
            var2=size(etiquetas_labels_predichas_matrix);
            %etiquetas_labels_predichas_matrix(:,conta)=etiquetas_labels_predichas_vector;
            
            for t1=var1+1:var2
                etiquetas_labels_predichas_vector(t1,1)=("N/A");
            end
            etiquetas_labels_predichas_matrix(:,conta)=etiquetas_labels_predichas_vector;
          
            
            %Este lazo completa el vector etiquetas_labels_predichas_vector de ser neceario
            %, ya que tiene q coincidir con la
            %dimension del vector etiquetas_labels_predichas_matrix para poder imprimirlo en cvs
%             if var1(1,1) == 36
%                 etiquetas_labels_predichas_vector(37,1)=("N/A");
%                 etiquetas_labels_predichas_vector(38,1)=("N/A");
%                 etiquetas_labels_predichas_vector(39,1)=("N/A");
%                 etiquetas_labels_predichas_vector(40,1)=("N/A");
%                 etiquetas_labels_predichas_vector(41,1)=("N/A");
%                 etiquetas_labels_predichas_vector(42,1)=("N/A");
%                 etiquetas_labels_predichas_matrix(:,conta)=etiquetas_labels_predichas_vector;
%             elseif var1(1,1) == 37
%                 etiquetas_labels_predichas_vector(38,1)=("N/A");
%                 etiquetas_labels_predichas_vector(39,1)=("N/A");
%                 etiquetas_labels_predichas_vector(40,1)=("N/A");
%                 etiquetas_labels_predichas_vector(41,1)=("N/A");
%                 etiquetas_labels_predichas_vector(42,1)=("N/A");
%                 etiquetas_labels_predichas_matrix(:,conta)=etiquetas_labels_predichas_vector;
%             elseif var1(1,1) == 38
%                 etiquetas_labels_predichas_vector(39,1)=("N/A");
%                 etiquetas_labels_predichas_vector(40,1)=("N/A");
%                 etiquetas_labels_predichas_vector(41,1)=("N/A");
%                 etiquetas_labels_predichas_vector(42,1)=("N/A");
%                 etiquetas_labels_predichas_matrix(:,conta)=etiquetas_labels_predichas_vector;
%             elseif var1(1,1) == 39
%                 etiquetas_labels_predichas_vector(40,1)=("N/A");
%                 etiquetas_labels_predichas_vector(41,1)=("N/A");
%                 etiquetas_labels_predichas_vector(42,1)=("N/A");
%                 etiquetas_labels_predichas_matrix(:,conta)=etiquetas_labels_predichas_vector;
%             elseif var1(1,1) == 40
%                 etiquetas_labels_predichas_vector(41,1)=("N/A");
%                 etiquetas_labels_predichas_vector(42,1)=("N/A");
%                 etiquetas_labels_predichas_matrix(:,conta)=etiquetas_labels_predichas_vector;
%             elseif var1(1,1) == 41
%                 etiquetas_labels_predichas_vector(42,1)=("N/A");
%                 etiquetas_labels_predichas_matrix(:,conta)=etiquetas_labels_predichas_vector;
%             end
            assignin('base','etiquetas_labels_predichas_matrix',etiquetas_labels_predichas_matrix);
            
            etiquetas_GT_vector(1,conta)=string(gestureName_GT); %
            %%%%CAMBIODANNYINICIO%%%%
            if verbose
                disp(gestureName_GT)
            end
            %%%%CAMBIODANNYFIN%%%%
            assignin('base','etiquetas_GT_vector',etiquetas_GT_vector);
            
            conta=conta+1;
            %disp(conta)
            %disp('etiquetas_labels_predichas_vector_without_NoGesture');
            %disp(etiquetas_labels_predichas_vector_without_NoGesture); %[N,1] %vector Full
            assignin('base','etiquetas_labels_predichas_vector_without_NoGesture',etiquetas_labels_predichas_vector_without_NoGesture);
            %var2=size(etiquetas_labels_predichas_vector_without_NoGesture)
            %etiquetas_labels_predichas_matrix_without_NoGesture(:,conta)=etiquetas_labels_predichas_vector_without_NoGesture;
            
            
            
            %disp('etiquetas_labels_predichas_vector_simplif');
            %disp(etiquetas_labels_predichas_vector_simplif);    % [1,N]  ok listo
            assignin('base','etiquetas_labels_predichas_vector_simplif',etiquetas_labels_predichas_vector_simplif);
            %size(etiquetas_labels_predichas_vector_simplif)
            
            %disp('ProcessingTimes_vector');
            %disp(ProcessingTimes_vector);    %[1,N] ok listo
            assignin('base','ProcessingTimes_vector',ProcessingTimes_vector);
            %size(ProcessingTimes_vector)
            
            %disp('TimePoints_vector');
            %disp(TimePoints_vector);         %[1,N] ok listo
            assignin('base','TimePoints_vector',TimePoints_vector);
            %size(TimePoints_vector)
            
            %             disp('class_result');
            %             disp(class_result);              %[1,1] ok listo
            assignin('base','class_result',class_result);
            %size(class_result)
            %------------------------------------------------------------------
            
                        %------------------------------------------------------------------
            %---  POST - Processing: elimino etiquetas espuria usando la
            %moda diferente de no gesto para crear vector de resultados que
            %va a la etaoa de reconocimiento
            post_processing_result_vector_lables=etiquetas_labels_predichas_vector_simplif;
            dim_vect=size(etiquetas_labels_predichas_vector_simplif);
            for i=1:dim_vect(1,2)
                if etiquetas_labels_predichas_vector_simplif(1,i) ~= class_result && etiquetas_labels_predichas_vector_simplif(1,i) ~= "noGesture"
                    post_processing_result_vector_lables(1,i)=class_result;
                else
                end  
            end
            assignin('base','post_processing_result_vector_lables',post_processing_result_vector_lables);
            %-------------------------------------------------------------
            %%%%CAMBIODANNYINICIO%%%%
            if verbose
               disp('Eval Recognition'); 
            end
            %%%%CAMBIODANNYFIN%%%%
            
            % GROUND TRUTH (no depende del modelo)------------
            repInfo.gestureName =  gestureName_GT; % OK -----  categorical({'waveIn'});   %CAMBIAR - poner etiqueta de muestra de señal
            assignin('base','gestureName_GT',gestureName_GT);
            repInfo.groundTruth = groundTruth_GT; %   REV -----
            assignin('base','groundTruth_GT',groundTruth_GT);
            %repInfo.groundTruth = false(1, 1000);   %Each_complete_signal;           %false(1, 1000);            %CAMBIAR
            %repInfo.groundTruth(800:1600) = true;   %CAMBIAR (64datos*40ventanas)
            
            %plot(repInfo.groundTruth)
            
            % PREDICCION--------------------------------------
            
            if post_processing == true
                response.vectorOfLabels = categorical(post_processing_result_vector_lables); % OK ----- [1,N] % categorical(etiquetas_labels_predichas_vector_simplif); % %CAMBIAR
            else
                response.vectorOfLabels = categorical(etiquetas_labels_predichas_vector_simplif); % OK ----- [1,N] % categorical(etiquetas_labels_predichas_vector_simplif); % %CAMBIAR
            end
            
            %response.vectorOfLabels = categorical(etiquetas_labels_predichas_vector_simplif); % OK ----- [1,N] % categorical(etiquetas_labels_predichas_vector_simplif); % %CAMBIAR
            response.vectorOfTimePoints = TimePoints_vector; % OK -----  [40 200 400 600 800 999]; %1xw double  TimePoints_vector                %CAMBIAR
            % tiempo de procesamiento
            response.vectorOfProcessingTimes = ProcessingTimes_vector; % OK -----[0.1 0.1 0.1 0.1 0.1 0.1]; % ProcessingTimes_vector'; % [0.1 0.1 0.1 0.1 0.1 0.1]; % 1xw double                                    %CAMBIAR
            response.class =  categorical(class_result); % OK ----- categorical({'waveIn'});                %aqui tengo que usar la moda probablemente           %CAMBIAR
            
            %-----------------------------------------------
            %r1 = 1;
            try
                r1 = evalRecognition(repInfo, response);
            catch
                warning('EL vector de predicciones esta compuesto por una misma etiqueta -> Func Eval Recog no funciona');
                r1.recogResult=0; fin_del_juego=1;
                if gestureName_GT==response.class
                    r1.classResult=1;
                else
                    r1.classResult=0;
                end
                
            end
            %assignin('base','r1',r1);
            
            if isempty(r1.recogResult) && fin_del_juego==1
                %Asigno recompensa en base al resultado de reconocimiento
                %disp('lazo1')
                %esto comentar si se requiere, solo si es no gesture se tiene esto
                %                 if fin_del_juego==1 && gestureName_GT==categorical({'noGesture'}) && class_result~="noGesture" %numIteration == maxIterationsAllowed % reward == -10  end the game - lose
                %                     %disp('lazo1-lost')
                %                     resultGame = 'lost';
                %                     disp('lost')
                %                     gameOn = false;
                %                     reward = -10;
                %                     % Suma_aciertos >= 30 &&
                %                 elseif  fin_del_juego==1 && gestureName_GT==categorical({'noGesture'}) && class_result=="noGesture" % numIteration == maxIterationsAllowed % eward == +10   end the game - win
                %                     %disp('lazo1-won')
                %                     resultGame = 'won ';
                %                     disp('won')
                %                     countWins = countWins + 1;
                %                     gameOn = false;
                %                     reward = +10;
                %
                %                 end
                
            else
                %disp('lazo2')
                if  r1.recogResult~=1 && fin_del_juego==1 %numIteration == maxIterationsAllowed % reward == -10  end the game - lose
                    resultGame = 'lost';
                    %%%%CAMBIODANNYINICIO%%%%
                    if verbose
                        disp('lost')
                    end
                    %%%%CAMBIODANNYFIN%%%%
                    gameOn = false;
                    reward = -10;  %-10
                    countLoses = countLoses +1;
                    % Suma_aciertos >= 30 &&
                elseif  r1.recogResult==1 && fin_del_juego==1 % numIteration == maxIterationsAllowed % eward == +10   end the game - win
                    resultGame = 'won ';
                    %%%%CAMBIODANNYINICIO%%%%
                    if verbose
                        disp('won')
                    end
                    %%%%CAMBIODANNYFIN%%%%
                    countWins = countWins + 1;
                    gameOn = false;
                    reward = +10; %+10
                    
                end
                
            end
            
        end
        
        % %$$$     $AQUI DEFINO LA RECOMPENSA EN FUNCION DE SI ES EL ESTADO FINAL    $$$$$
        %         compare_vector= etiquetas==acciones_predichas_vector;  % COMPARO ETIQUETAS PREDICHAS CON GROUNDTRUTH
        %         Suma_aciertos=sum(compare_vector(:) == 1);
        %
        %         %AQUI CAMBIAR PORCENTAJE DE ACIERTOS PARA CONSIDERAR QUE UN JUEGO
        %         %(PREDICCION DE GESTO EN UNA SECUENCA DE VENTANAS DE EMG) ES EXITOSO O NO
        %         if  Suma_aciertos < 30 && numIteration == maxIterationsAllowed % reward == -10  end the game - lose
        %             resultGame = 'lost';
        %             disp('lost')
        %             gameOn = false;
        %
        %         elseif  Suma_aciertos >= 30 && numIteration == maxIterationsAllowed % eward == +10   end the game - win
        %             resultGame = 'won ';
        %             disp('won')
        %             countWins = countWins + 1;
        %             gameOn = false;
        %         end
        
        %$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        %----------------------------------------------------------------------
        %$$$     $AQUI DEFINO LA RECOMPENSA EN FUNCION DE SI ES EL ESTADO FINAL    $$$$$
        %         compare_vector= etiquetas==acciones_predichas_vector;  % COMPARO ETIQUETAS PREDICHAS CON GROUNDTRUTH
        %         Suma_aciertos=sum(compare_vector(:) == 1);
        
        %AQUI CAMBIAR PORCENTAJE DE ACIERTOS PARA CONSIDERAR QUE UN JUEGO
        %(PREDICCION DE GESTO EN UNA SECUENCA DE VENTANAS DE EMG) ES EXITOSO O NO
        %    Suma_aciertos < 30 &&
        %         if  Suma_aciertos < 10 && fin_del_juego==1 %numIteration == maxIterationsAllowed % reward == -10  end the game - lose
        %             resultGame = 'lost';
        %             disp('lost')
        %             gameOn = false;
        %             reward = -10;
        %             % Suma_aciertos >= 30 &&
        %         elseif  Suma_aciertos >= 10 && fin_del_juego==1 % numIteration == maxIterationsAllowed % eward == +10   end the game - win
        %             resultGame = 'won ';
        %             disp('won')
        %             countWins = countWins + 1;
        %             gameOn = false;
        %             reward = +10;
        %         end
        %-----------------------------------------------------------
        %$$$$$$$Aqui abajo esta la recompensa en funcion del resultado de reconocimiento$$$$$$$$$
        
        %         if  r1.recogResult~=1 && fin_del_juego==1 %numIteration == maxIterationsAllowed % reward == -10  end the game - lose
        %             resultGame = 'lost';
        %             disp('lost')
        %             gameOn = false;
        %             reward = -10;
        %             % Suma_aciertos >= 30 &&
        %         elseif  r1.recogResult==1 && fin_del_juego==1 % numIteration == maxIterationsAllowed % eward == +10   end the game - win
        %             resultGame = 'won ';
        %             disp('won')
        %             countWins = countWins + 1;
        %             gameOn = false;
        %             reward = +10;
        %         %esto comentar si se requiere, solo si es no gesture se tiene esto
        %         elseif fin_del_juego==1 && gestureName_GT=='noGesture' && class_result~='noGesture' %numIteration == maxIterationsAllowed % reward == -10  end the game - lose
        %             resultGame = 'lost';
        %             disp('lost')
        %             gameOn = false;
        %             reward = -10;
        %             % Suma_aciertos >= 30 &&
        %         elseif  fin_del_juego==1 && gestureName_GT=='noGesture' && class_result=='noGesture' % numIteration == maxIterationsAllowed % eward == +10   end the game - win
        %             resultGame = 'won ';
        %             disp('won')
        %             countWins = countWins + 1;
        %             gameOn = false;
        %             reward = +10;
        %
        %         end
        
        
        
        
        %$$$$$$$$$$ AQUI DEFINO LA RECOMPENSA EN FUNCION DE SI ES EL ESTADO FINAL $$$$$$$$$$$$$
        %         if  numIteration ~= Numero_Ventanas_GT-1 % maxWindowsAllowed  % abs(reward) ~= 10
        %             % Target for a non-terminal state  (or learned value)
        %             %Ecuacion de funcion de valor Q(s)=R(t+1)+gamma*maxQvalue'
        %             target = reward + gamma*maxQval;    %valor de tarjet si aun no termina episodio
        %             %disp('target = reward + gamma*maxQval;')
        %         else
        %             % Taget for a terminal state
        %             target = reward;                    %valor de tarjet para fin de episodio
        %             %disp('target = reward;')
        %         end
        %$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        
        
        %******************************************************
        %         % Data for training the ANN     -     % valores de Q y de estados dentro de cada episodio se guardan en estas variables
        %         dataX(numIteration, :) = state(:)';
        %         dataY(numIteration, :) = old_Qval;
        %         dataY(numIteration, action) = target;
        %         % Updating the state
        %         state = new_state;
        %*****************************************************
        
        %         % Data for training the ANN     -     % valores de Q y de estados dentro de cada episodio se guardan en estas variables
        %         dataX(numIteration, :) = state;       % CAMBIE SOLO AQUI
        %         dataY(numIteration, :) = old_Qval;
        %         dataY(numIteration, action) = target;
        
        % Cumulative reward so far for the current episode
        cumulativeGameReward = cumulativeGameReward + reward;
        
        % Cumulative reward so far for the current episode
        cumulativeIterationReward = cumulativeIterationReward + reward;
        
        % Updating the state
        state = new_state;
        
    end
    
    % Sampling randomly the data from the experience replay buffer
    if numIterationTotal > ceil( 1.05*lengthBuffer )
        %disp('sampling')
        [dummy, idx] = sort(rand(lengthBuffer, 1));
        randIdx = idx(1:miniBatchSize);
        dataX = zeros(miniBatchSize, state_length);  %64
        dataY = zeros(miniBatchSize, 6);
        % Computations for the minibatch
        for numExample = 1:miniBatchSize
            % Getting the value of Q(s, a)
            old_state_er = gameReplay(randIdx(numExample), 1:state_length); %64
            [dummyVar, A] = forwardPropagation(old_state_er, weights,... %old_state_er(:)'
                transferFunctions, options);
            old_Qval_er = A{end}(:, 2:end);
            % Getting the value of max_a_Q(s',a')
            new_state_er = gameReplay(randIdx(numExample), (end - (state_length-1)):end);  %63
            [dummyVar, A] = forwardPropagation(new_state_er, weights,... %new_state_er(:)'
                transferFunctions, options);
            new_Qval_er = A{end}(:, 2:end);
            maxQval_er = max(new_Qval_er);
            action_er = gameReplay(randIdx(numExample), state_length+1);           %65
            reward_er = gameReplay(randIdx(numExample), state_length+2);           %66
            %                 if reward_er == 0  %-1
            %                     % Non-terminal state
            %                     update_er = reward_er + gamma*maxQval_er;
            %                 else
            %                     % Terminal state
            %                     update_er = reward_er;
            %                 end
            
            if Reward_type ==true
                
                if reward_er == -1  %-1       OJO CON ESTE
                    % Non-terminal state
                    update_er = reward_er + gamma*maxQval_er;
                else
                    % Terminal state
                    update_er = reward_er;
                end
            else
                
                if reward_er == 0  %-1       OJO CON ESTE
                    % Non-terminal state
                    update_er = reward_er + gamma*maxQval_er;
                else
                    % Terminal state
                    update_er = reward_er;
                end
            end
            
            
            % Data for training the ANN
            dataX(numExample, :) = old_state_er;  %old_state_er(:)'
            dataY(numExample, :) = old_Qval_er;
            dataY(numExample, action_er) = update_er;
        end
        % Updating the weights of the ANN
        [cost(epoch), gradient] = regressionNNCostFunction(dataX, dataY,...
            numNeuronsLayers,...
            theta,...
            transferFunctions,...
            options);
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
    end
    
    % Cumulative reward of all the past episodes
    cumulativeEpochReward = cumulativeEpochReward + cumulativeIterationReward;
    % History of the rewards for each episode
    averageEpochReward(epoch) = cumulativeEpochReward/epoch;
    % History of the average maxQval_er value for the test set of states
    [dummyVar, A] = forwardPropagation(testStates, weights, transferFunctions, options);
    test_Qval = A{end}(:, 2:end);
    averageMaxQvalue(epoch) = mean( max(test_Qval, [], 2) );
    % History of average of wins per epoch
    historyAverageWins(epoch) = countWins/epoch;
    
    %%%%CAMBIODANNYINICIO%%%%
    if verbose
        fprintf('Epoch: %d of %d, cost = %3.2f, result = %s, epsilon = %1.2f, Qval = [%3.2f, %3.2f, %3.2f, %3.2f, %3.2f, %3.2f] \n',...
            epoch, numEpochs, cost(epoch), resultGame, epsilon, Qval(1), Qval(2), Qval(3), Qval(4), Qval(5), Qval(6));
        disp('class_predicted'); disp(class_result);              %[1,1] ok listo
    end
    %%%%CAMBIODANNYFIN%%%%
    
    % Annealing the epsilon
    if epsilon > 0.10
        epsilon = 0.2; %epsilon0*exp(-7*epoch/numEpochs);
    end
    
    
    
    
    %     %------------------%FIN del episodio-----------------------------------
    %     %-----------------------------------------------------------------------
    %     dataXN = dataX(1:numIteration, :);        %guardo datos de S y Q de 1 episodio en vectores
    %     %disp(dataXN)
    %     dataYN = dataY(1:numIteration, :);        %guardo datos de S y Q de 1 episodio en vectores
    %     %disp(dataYN)
    %     % Updating the weights                    % ACTUALIZO PESOS DE la red neuronal
    %     disp('Updating the weights')
    %     [cost(epoch), gradient] = regressionNNCostFunction(dataXN, dataYN,...
    %         numNeuronsLayers,...
    %         theta,...
    %         transferFunctions,...
    %         options);
    %     % Updating the weights of the ANN - momentum ----------------------------
    %     if strcmp(options.typeUpdate, 'momentum')                %  Momentum update
    %         % Increase momentum after momIncrease iterations
    %         if epoch == numEpochsToIncreaseMomentum
    %             momentum = options.momentum;
    %         end
    %         velocity = momentum*velocity + alpha*gradient;
    %         theta = theta - velocity;
    %         % Annealing the learning rate
    %         alpha = learningRate*exp(-5*epoch/numEpochs);
    %     else
    %         error('Invalid selection for updating the weights \n');
    %     end
    %
    %     % Cumulative reward of all the past episodes
    %     cumulativeEpochReward = cumulativeEpochReward + cumulativeGameReward;
    %     % History of the rewards for each episode
    %     averageEpochReward(epoch) = cumulativeEpochReward/epoch;
    %     % History of the average maxQval_er value for the test set of states
    %     [dummyVar, A] = forwardPropagation(testStates, weights, transferFunctions, options);
    %     test_Qval = A{end}(:, 2:end);
    %     averageMaxQvalue(epoch) = mean( max(test_Qval, [], 2) );
    %     % History of average of wins per epoch
    %     historyAverageWins(epoch) = countWins/epoch;
    %
    %     %$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    %     fprintf('Epoch: %d of %d, cost = %3.2f, result = %s, epsilon = %1.2f, Qval = [%3.2f, %3.2f, %3.2f, %3.2f, %3.2f, %3.2f] \n',...
    %         epoch, numEpochs, cost(epoch), resultGame, epsilon, old_Qval(1), old_Qval(2), old_Qval(3), old_Qval(4),...
    %         old_Qval(5), old_Qval(6));   %AQUI CAMBIAR CON EMG GESTOS
    %     %$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    %
    %     % Annealing the epsilon
    %     if epsilon > 0.10
    %         epsilon = epsilon0*exp(-7*epoch/numEpochs); %1;
    %     end
end

n1=size(class_result_vector);

for nk=1:n1(1,2)
    if etiquetas_GT_vector(1,nk)==class_result_vector(1,nk)
        number_classif_ok=number_classif_ok+1;
    else
        number_classif_failed=number_classif_failed+1;
    end
end

%%%%CAMBIODANNYINICIO%%%%
if verbose
    disp('TRAINING RESULTS')
    fprintf('Count Wins Recog = %d --- Count Loses recog = %d --- Total Recog = %d  \n', countWins, countLoses,countWins+countLoses);
    fprintf('Count Wins Classif = %d --- Count Loses classig = %d --- Total Classif = %d \n', number_classif_ok, number_classif_failed,number_classif_ok+number_classif_failed);
    fprintf('Count Wins per window = %d --- Count Loses per window = %d --- Total Classif per window = %d \n', countWins2, countLoses2,countWins2+countLoses2 );
else
    summary = [countWins, number_classif_ok, countWins2, countLoses, number_classif_failed, countLoses2];
end
%%%%CAMBIODANNYFIN%%%%

assignin('base','etiquetas_labels_predichas_matrix',etiquetas_labels_predichas_matrix);
%cell2csv('new_cell2csv.csv', etiquetas_labels_predichas_matrix)
%csvwrite('etiquetas_labels_predichas_matrix.txt',etiquetas_labels_predichas_matrix)
%csvwrite('etiquetas_labels_predichas_matrix_without_NoGesture.txt',etiquetas_labels_predichas_matrix_without_NoGesture)
Full_train_data=[etiquetas_labels_predichas_matrix;etiquetas_GT_vector;class_result_vector];
%cell2csv('Full_test_dataTRAINING.csv', Full_train_data)
s1 = 'Full_test_dataTRAINING_';
s2   = evalin('base', 's2');
s3 = '.csv';

%%%%CAMBIODANNYINICIO%%%%
s = strcat("results/csv_full_test_data/", s1,s2,s3);
%%%%CAMBIODANNYFIN%%%%

cell2csv(s, Full_train_data)
%------------^^^^^^for epoch = 1:numEpochs^^^^^^ -----------------------------------------------------------
%-----------------------------------------------------------------------
%-----------------------------------------------------------------------
%-----------------------------------------------------------------------

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
a1 = 'Fig1_Exp_';
a2   = evalin('base', 's2');
a3 = '.png';

%%%%CAMBIODANNYINICIO%%%%
s = strcat("results/figures/", a1,a2,a3);
%%%%CAMBIODANNYFIN%%%%


saveas(gcf,s)
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
a1 = 'Fig2_Exp_';
a2   = evalin('base', 's2');
a3 = '.png';


%%%%CAMBIODANNYINICIO%%%%
s = strcat("results/figures/", a1,a2,a3);
%%%%CAMBIODANNYFIN%%%%


saveas(gcf,s)
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
a1 = 'Fig3_Exp_';
a2   = evalin('base', 's2');
a3 = '.png';


%%%%CAMBIODANNYINICIO%%%%
s = strcat("results/figures/", a1,a2,a3);
%%%%CAMBIODANNYFIN%%%%


saveas(gcf,s)
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
a1 = 'Fig4_Exp_';
a2   = evalin('base', 's2');
a3 = '.png';


%%%%CAMBIODANNYINICIO%%%%
s = strcat("results/figures/", a1,a2,a3);
%%%%CAMBIODANNYFIN%%%%


saveas(gcf,s)
% Reshaping the weights

%%%%CAMBIODANNYINICIO%%%%
if ~verbose
   close all; 
end
%%%%CAMBIODANNYFIN%%%%

weights = reshapeWeights(theta, numNeuronsLayers);
end