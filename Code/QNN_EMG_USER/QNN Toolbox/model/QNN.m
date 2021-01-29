classdef QNN < handle
    
    properties
        qnnOption;
        Reward_type;  % on si quiero recompensa -1 x ventana (clasif) y -10 x recog
        amount_reward_correct = 1;
        amount_reward_incorrect = -1;
        number_gestures_taken_from_user;  % This is RepTraining, the same
        theta;
        exp_replay_lengthBuffer;
        % gameReplay -> exp replay with dims: exp_replay_lengthBufferx82
        % 82 = 1 reward, 1 action, 40 of length s, 40 of length of new s'
        % each state is the vector of 40 elemnts of feature extractor
        gameReplay;
        total_num_windows_predicted = 0;
    end
    
    methods
        function obj = QNN(qnnOption, Reward_type, number_gestures_taken_from_user)
            %QNN Construct an instance of this class
            obj.qnnOption = qnnOption;
            obj.Reward_type = Reward_type;
            obj.number_gestures_taken_from_user = number_gestures_taken_from_user;
            obj.theta = randInitializeWeights(qnnOption.numNeuronsLayers);
            % Experience replay initialization ------------------------------
            obj.exp_replay_lengthBuffer = ceil(3*qnnOption.miniBatchSize);  % ???? por qué 3
            s  =  nan(obj.exp_replay_lengthBuffer, 40);  % 40 because feature extract
            sp =  nan(obj.exp_replay_lengthBuffer, 40);
            a  =  nan(obj.exp_replay_lengthBuffer, 1);
            r  =  nan(obj.exp_replay_lengthBuffer, 1);
            obj.gameReplay = [s, a, r, sp]; %(state, action, reward, state')
            
            
        end
        
        function [weights, summary] = train(this, verbose_level)
            summary = [];
            EMG_window_size = evalin('base', 'WindowsSize');                                                %AQUI PONER WINDOW SIZE
            Stride = evalin('base', 'Stride');
            % num_windows = 1 +
            % ceil((numberPointsUserEMG-EMG_window_size)/Stride)
            % or floor((numberPointsUserEMG-EMG_window_size)/Stride) to
            % ignore the last part incomplete of the signal
            
            
            
            orientation      = evalin('base', 'orientation');
            post_processing      =  evalin('base', 'post_processing');
             
            % dataPacketSize is the amount of users in "Data/Specific" including "." and ".."
            % so we should not include  them with dataPacketSize1-2
            dataPacketSize     = evalin('base', 'dataPacketSize');

            % here, numEpochs is the number of gestures of all users in sum.
            % by example: 3 users with 5 gestures each one for training, the result is 5*((3+2)-2)= 15  
            % in QNN each gesture/"epoch" can be interpreted as an episode of num_windows
            repTotalTraining  = this.number_gestures_taken_from_user*(dataPacketSize-2);  % is the same of repTotalTraining
            
            
            % Setup for momentum update of the weights of the ANN
            momentum = this.qnnOption.initialMomentum;
            velocity = zeros( size(this.theta) );
            
            % Training the ANN
            cost = zeros(1, repTotalTraining);
            
            % Initializing the history of the average reward
            averageEpochReward = zeros(1, repTotalTraining);                           
            
            
            averageMaxQvalue = zeros(1, repTotalTraining);
            historyAverageWins = zeros(1, repTotalTraining);

            
            numIterationTotal = 0;
            
            conta=1;

            
            
            debug_step = ceil(this.number_gestures_taken_from_user/5); %%% Just for debugging
            
            dataPacket = evalin('base','dataPacket');
                
            % for each user in Specific
            for data_user=1:dataPacketSize-2
                
                 %%%% PENDING
                % num_windows = 1 + floor((1000-EMG_window_size)/Stride);  % num_windows = NumeroVentanasGT
                % numTestingSamples = max(0.1*(num_windows+1), 20);  % numTestingSamples = max(0.1*num_windows, 20);   ????????
                % testStates = zeros(numTestingSamples, 40);  % ???????????????????????????????????????
                % etiquetas_labels_predichas_matrix=strings(num_windows,numEpochs);
            
                userIndex   = evalin('base', 'userIndex');
                userData = loadSpecificUser(dataPacket, userIndex);
                energy_index = strcmp(orientation(:,1), userData.userInfo.name);
                rand_data=orientation{energy_index,6};
                
                cumulativeUserReward = 0;  % Inicializo reward acumulado
                
                % Each episode is a gesture taken from user
                % each episode has num_windows with the window_size/stride
                for episode=1:this.number_gestures_taken_from_user
                    
                    if mod(episode, debug_step) == 0 && verbose_level >= 1 || episode == 1 || episode == this.number_gestures_taken_from_user
                        fprintf("User: %s, Episode(Gesture) %d of %d\n", userData.userInfo.name, episode, this.number_gestures_taken_from_user);
                    end
                   
                    
                    emgRepetition = evalin('base','emgRepetition');
                    numberPointsEmgData = length(userData.training{rand_data(emgRepetition),1}.emg);
                    num_windows = getNumberWindows(numberPointsEmgData, EMG_window_size, Stride, false);
                    groundTruthIndex = userData.training{rand_data(emgRepetition),1}.groundTruthIndex;
                    gestureName = userData.training{rand_data(emgRepetition),1}.gestureName;
                    
                    this.executeEpisode(num_windows, orientation, dataPacketSize, ...
                                        repTotalTraining, EMG_window_size, Stride, ...
                                        groundTruthIndex, gestureName, ...
                                        verbose_level-1);
                    
                    % this variable is a reference for the exp replay buffer
                    this.total_num_windows_predicted = this.total_num_windows_predicted + num_windows;

                end
                
            end

            
            
        end
        
        function executeEpisode(this, num_windows, orientation, dataPacketSize, ...
                                repTotalTraining, EMG_window_size, Stride, ...
                                groundTruthIndex, gestureName, ...
                                verbose_level)
            % -----  El juego del episodio viene a ser el barrido de ventanas en una muestra EMG
            % ----- Se gana el juego si el resultado de reconocimiento es correctom
            % ----- sino se pierde el juego y se tiene una penalizacion
            
            %Creo vector de etiquetas para Ground truth x ventana
            % PENDING: revisar y mejorar, aqui puede estar la mayor concentración de problema
            gt_gestures_labels = mapGroundTruthToLabels(num_windows, EMG_window_size, Stride, groundTruthIndex, gestureName);
            %Creo vector de etiquetas de Ground truth con valores numericos
            gt_gestures_labels_num = mapGestureLabelsToNumbers(num_windows, gt_gestures_labels);
            
            %---- Obtengo datos de primera ventana de cada muestra e inicializo vectores de ground truth-----------------*

            Vector_EMG_Tiempos_GT=zeros(1,num_windows); %creo vector de tiempos de gt
            Vector_EMG_Puntos_GT=zeros(1,num_windows);  %creo vector de puntos de gt
            
            % ---- Inicializo variables requeridas para guardar datos de prediccion
            % - Quito primera etiqueta de ventana ya que esa no se predice en el
            % lazo while
            etiquetas = gt_gestures_labels_num; %1+round((5)*rand(Numero_Ventanas_GT,1));   %1+round((5)*rand(maxWindowsAllowed,1)); %%%%%%%% AQUI PONER ground truth de cada ventana EMG - gestos de 1 a 6
            etiquetas_labels_predichas_vector=strings;
            %etiquetas_labels_predichas_matrix=strings;
            etiquetas_labels_predichas_vector_without_NoGesture=strings;
            
            % every episode will have a prediction for each window
            acciones_predichas_vector = zeros(num_windows,1);

            % ---- inicializo parametros medicion tiempo, y vectores de prediccion
            % necesarios para evaluar reconocimiento en cada epoca -----------------
            ProcessingTimes_vector=[];
            TimePoints_vector=[];
            n1=0;
            etiquetas_labels_predichas_vector_simplif=strings;
            
            count_wins_in_episode = 0;
            
            tic;
            
            window_n = 1;
            % Its necesary get the first state for replay exp
            [~,~,Features_GT,Tiempos_GT,Puntos_GT, ~, ~, ~, groundTruth_GT] = ...
                Code_1(orientation,dataPacketSize, this.number_gestures_taken_from_user, repTotalTraining, verbose_level-1);

            Vector_EMG_Tiempos_GT(1,window_n)=Tiempos_GT;      %copio primer valor en vector de tiempos de gt
            Vector_EMG_Puntos_GT(1,window_n)=Puntos_GT;        %copio primer valor en vector de tiempos de gt

            %---- Defino estado en base a cada ventana EMG
            % window_state is not related with the S or S'
            % because dowsnt matter what action is taken THE NEXT WINDOW STATE WILL BE THE SAME
            
            window_state = table2array(Features_GT);
            state = 0;  % the objective is take the steps correct to increase state from 0 to num_windows in this episode
            
            while window_n < num_windows
                
                [Qval, action] = this.selectAction(state);
                
                % AQUI SE VAN GUARDADNO LAS ACCIONES PREDICHAS DENTRO DEl vector de UNA EPOCA
                acciones_predichas_vector(window_n,1)=action;
                etiquetas_labels_predichas_vector(window_n,1)=QNN.convertActionIndexToLabel(action);
                
                % WO    = 1 % WI    = 2 % FIST  = 3 % OPEN  = 4 % PINCH = 5 % RELAX = 6
                
                real_action=gt_gestures_labels_num(window_n);
                reward_for_state = this.calculateReward(action, real_action, this.Reward_type);
                
                if reward_for_state == this.amount_reward_correct
                    count_wins_in_episode = count_wins_in_episode + 1;
                end
                
                if verbose_level >=1 && reward_for_state~=0
                    fprintf("reward for state %d of %d in actual episode = %d\n", window_n, num_windows, reward_for_state);
                end
                
                %---- Experience replay storage ------------------------------------
                idx = mod(this.total_num_windows_predicted + window_n, this.exp_replay_lengthBuffer);
                if idx == 0
                    idx = this.exp_replay_lengthBuffer;
                end
                % the rows of game Replay are 3*minBatchSize
                
                %Leo datos de muestras
                [~,~,Features_GT,Tiempos_GT,Puntos_GT, ~, ~, ~, ~] = ...
                    Code_1(orientation,dataPacketSize, this.number_gestures_taken_from_user, repTotalTraining, verbose_level-1);

                Vector_EMG_Tiempos_GT(1,window_n+1)=Tiempos_GT;      %copio primer valor en vector de tiempos de gt
                Vector_EMG_Puntos_GT(1,window_n+1)=Puntos_GT;        %copio primer valor en vector de tiempos de gt
                
                %---- The next state pre defined
                new_state = table2array(Features_GT);
                
                this.gameReplay(idx, :) = [state, action, reward_for_state, new_state];   %[state(:)', action, reward, new_state(:)'];
                %---------------------------------------------------------------

                

                %Acondicionar vectores - si el signo anterior no es igual al signo acual entocnes mido tiempo
                if window_n~=num_windows-1 && ...
                        etiquetas_labels_predichas_vector(window_n,1) ~= etiquetas_labels_predichas_vector(window_n-1,1)

                    n1=n1+1;
                    ProcessingTimes_vector(1,n1) = toc;  %mido tiempo transcurrido desde ultimo cambio de gesto
                    tic;

                    %obtengo solo etiqueta que se ha venido repetiendo hasta instante numIteration-1
                    etiquetas_labels_predichas_vector_simplif(1,n1)=etiquetas_labels_predichas_vector(window_n-1,1);

                    %obtengo nuevo dato para vector de tiempos
                    TimePoints_vector(1,n1)=Stride*window_n+EMG_window_size/2;           %necesito dato de stride y tamaño de ventana de Victor

                elseif window_n == num_windows-1 %==maxWindowsAllowed    % si proceso la ultima ventana de la muestra de señal EMG
                    n1=n1+1;
                    ProcessingTimes_vector(1,n1) = toc;  %mido tiempo transcurrido desde ultimo cambio de gesto
                    tic;

                    %obtengo solo etiqueta que no se ha repetido hasta instante numIteration-1
                    etiquetas_labels_predichas_vector_simplif(1,n1)=etiquetas_labels_predichas_vector(window_n,1);

                    %obtengo dato final para vector de tiempos
                    kj=size(groundTruth_GT);  %  se supone q son 1000 puntos
                    TimePoints_vector(1,n1)=  kj(1,2);                 %AQUI CAMBIAR  %1000 puntos

                end

                %Saco la moda de los gestos diferentes a NoGesture
                temp1=(size(etiquetas_labels_predichas_vector));
                temp1=temp1(1,1);

                %saco no gesture de este vector para poder usar funcion moda
                %-------- REVISAR-ESTO HABILITAR SI REQUIERO QUITAR NO GESTURE -----
                for i=1:temp1
                    if etiquetas_labels_predichas_vector(i,1)~="noGesture"
                        etiquetas_labels_predichas_vector_without_NoGesture(i,1)=etiquetas_labels_predichas_vector(i,1);
                    else
                    end
                end

                %dependiendo variable "noGestureDetection" Saco moda 1) sin considerar no gesto ,o 2) considerando no gesto
                noGestureDetection   = evalin('base', 'noGestureDetection');
                if noGestureDetection
                    class_result=mode(categorical(etiquetas_labels_predichas_vector));    %Saco moda incluyendo etiqueta de NoGesture
                else
                    class_result=mode(categorical(etiquetas_labels_predichas_vector_without_NoGesture));  %Saco la moda de las etiquetas dif a no gesture
                end

                %Si al sacar la moda todo es no gesto (<missing>), entonces la moda es no gesto
                if ismissing(class_result)
                    class_result="noGesture";
                end

                %esto es para guardar resultado de clasificacion 
                class_result_vector(1,conta)=string(class_result); %
                assignin('base','class_result_vector',class_result_vector);


                % Cumulative reward so far for the current episode
                %cumulativeGameReward = cumulativeGameReward + reward;        % recompensa acumulada
                %-----------------------------
                %%%%%%%%%%%%%%%%%%%%%%%% Configurar cuanto estaré en un estado
                %%%%%%%%%%%%%%%%%%%%%%%% terminal (depende de numero de ventanas)


                %$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                %AQUI EVALUO EL RECONOCIMIENTO UNA VEZ QUE SE TERMINA EL EPISODIO

                if  numIteration == Numero_Ventanas_GT %==maxWindowsAllowed % reward == -10  end the game - lose

                    %-----------check de las variables predichas que entran a eval de reconocimiento
                    assignin('base','etiquetas_labels_predichas_vector',etiquetas_labels_predichas_vector);
                    var1=size(etiquetas_labels_predichas_vector);
                    var2=size(etiquetas_labels_predichas_matrix);
                    %etiquetas_labels_predichas_matrix(:,conta)=etiquetas_labels_predichas_vector;

                    for t1=var1+1:var2
                        etiquetas_labels_predichas_vector(t1,1)=("N/A");
                    end
                    etiquetas_labels_predichas_matrix(:,conta)=etiquetas_labels_predichas_vector;

                    assignin('base','etiquetas_labels_predichas_matrix',etiquetas_labels_predichas_matrix);

                    etiquetas_GT_vector(1,conta)=string(gestureName_GT); %
                    %%%%CAMBIODANNYINICIO%%%%
                    if verbose_level >= 1
                        disp(gestureName_GT)
                    end
                    %%%%CAMBIODANNYFIN%%%%
                    assignin('base','etiquetas_GT_vector',etiquetas_GT_vector);

                    conta=conta+1;
                    assignin('base','etiquetas_labels_predichas_vector_without_NoGesture',etiquetas_labels_predichas_vector_without_NoGesture);
                    %var2=size(etiquetas_labels_predichas_vector_without_NoGesture)
                    %etiquetas_labels_predichas_matrix_without_NoGesture(:,conta)=etiquetas_labels_predichas_vector_without_NoGesture;

                    assignin('base','etiquetas_labels_predichas_vector_simplif',etiquetas_labels_predichas_vector_simplif);
                    %size(etiquetas_labels_predichas_vector_simplif)

                    assignin('base','ProcessingTimes_vector',ProcessingTimes_vector);

                    assignin('base','TimePoints_vector',TimePoints_vector);
                    assignin('base','class_result',class_result);
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
                    if verbose_level >= 1
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

                % Cumulative reward so far for the current episode
                cumulativeGameReward = cumulativeGameReward + reward;

                % Cumulative reward so far for the current episode
                cumulativeIterationReward = cumulativeIterationReward + reward;

                
                
                state = new_state;
                window_n = window_n + 1;
            end
            

            % Sampling randomly the data from the experience replay buffer
            if numIterationTotal > ceil( 1.05*lengthBuffer )
                %disp('sampling')
                [dummy, idx] = sort(rand(lengthBuffer, 1));
                randIdx = idx(1:miniBatchSize);
                dataX = zeros(miniBatchSize, 40);  %64
                dataY = zeros(miniBatchSize, 6);
                % Computations for the minibatch
                for numExample = 1:miniBatchSize
                    % Getting the value of Q(s, a)
                    old_state_er = gameReplay(randIdx(numExample), 1:40); %64
                    [dummyVar, A] = forwardPropagation(old_state_er, weights,... %old_state_er(:)'
                        transferFunctions, options);
                    old_Qval_er = A{end}(:, 2:end);
                    % Getting the value of max_a_Q(s',a')
                    new_state_er = gameReplay(randIdx(numExample), (end - 39):end);  %63
                    [dummyVar, A] = forwardPropagation(new_state_er, weights,... %new_state_er(:)'
                        transferFunctions, options);
                    new_Qval_er = A{end}(:, 2:end);
                    maxQval_er = max(new_Qval_er);
                    action_er = gameReplay(randIdx(numExample), 41);           %65
                    reward_er = gameReplay(randIdx(numExample), 42);           %66
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
        
        function [Qval, action_index] = selectAction(this, state)
            % Reshaping the weights of the ANN
            weights = reshapeWeights(this.theta, this.qnnOption.numNeuronsLayers);        % Reshaping the weights of the ANN (verificar shape)

            % Predicting the response of the ANN for the current state
            options.reluThresh = this.qnnOption.reluThresh;
            
            [~, A] = forwardPropagation(state, weights,...     % ANN to obtain update weights - state is a vector
                        this.qnnOption.transferFunctions, options);

            Qval = A{end}(:, 2:end);                                  % 6 Valores de Q - uno por cada accion

            [~, idx] = max(Qval);                              % obtengo indice de Qmax a partir de vector Qval


            % Epsilon-greedy action selection - REV Con epsilon=1
            % Inicialmente hace solo exploracion, luego el valor de epsilon se va reduciendo a medida que tengo mas informacion
            % Si rand <= epsilon, obtengo un Q de manera aleatoria, el cual será diferente a Qmax (exploracion)

            if rand <= this.qnnOption.epsilon            % siempre se cumple con epsilon=1% %Initial value of epsilon for the epsilon-greedy exploration
                full_action_list = 1:6;    %actionList = 1:6;               % posibles acciones  (AQUI HAY QUE CAMBIAR - #clases) usar gestos
                actionList = full_action_list(full_action_list ~= idx);           %  Crea lista con las acciones q no tienen Qmax
                idx_valid_action = randi([1 length(actionList)]);
                idx = full_action_list(actionList(idx_valid_action));
            end

            % - Predigo accion en base a Epsilon-Greedy
            action_index = idx;            
        end
        
        function reward = calculateReward(this, action, real_action, reward_type)
            
            if reward_type == 1
                if real_action == action
                    reward = this.amount_reward_correct;
                else
                    reward = this.amount_reward_incorrect;
                end
            else
                reward = 0;
            end
        end
        
        
        
        function [summary] = test(this, rangeDown, verbose_level)
            disp("Beginning testing");
            summary = 1:6;
        end
    end
    
    methods(Static)
        function action_text = convertActionIndexToLabel(action)
            gesture_names = ["waveOut", "waveIn", "fist", "open", "pinch", "noGesture"];
            action_text = gesture_names(action);                
        end 
    end
end

