classdef QNN < handle
    
    properties
        qnnOption;
        alpha;  % learning rate that change
        Reward_type;  % on si quiero recompensa -1 x ventana (clasif) y -10 x recog
        amount_reward_correct = 1;
        amount_reward_incorrect = -1;
        amount_reward_recognition_correct = 10;
        amount_reward_recognition_incorrect = -10;
        
        number_gestures_taken_from_user;  % This is RepTraining, the same
        theta;
        exp_replay_lengthBuffer;
        % gameReplay -> exp replay with dims: exp_replay_lengthBufferx4
        % 4 = 1 reward, 1 action, 40 of length s, 0 of length of new s'
        % each state is a numeric value is scalar
        gameReplay;
        total_num_windows_predicted = 0;
    end
    
    methods
        function obj = QNN(qnnOption, Reward_type, number_gestures_taken_from_user)
            %QNN Construct an instance of this class
            obj.qnnOption = qnnOption;
            obj.alpha = qnnOption.learningRate;
            
            obj.Reward_type = Reward_type;
            obj.number_gestures_taken_from_user = number_gestures_taken_from_user;
            obj.theta = randInitializeWeights(qnnOption.numNeuronsLayers);
            % Experience replay initialization ------------------------------
            obj.exp_replay_lengthBuffer = ceil(3*qnnOption.miniBatchSize);  % ???? por qué 3
            s  =  nan(obj.exp_replay_lengthBuffer, 40);  % 1 for optimization
            % sp =  nan(obj.exp_replay_lengthBuffer, 1);
            a  =  nan(obj.exp_replay_lengthBuffer, 1);
            r  =  nan(obj.exp_replay_lengthBuffer, 1);
            obj.gameReplay = [s, a, r]; %(state, action, reward, state') , sp
            
            
        end
        
        function [weights, summary, accuracy_by_episode] = train(this, verbose_level)
            summary = [];
            EMG_window_size = evalin('base', 'WindowsSize');                                                %AQUI PONER WINDOW SIZE
            Stride = evalin('base', 'Stride');
            
            orientation      = evalin('base', 'orientation');
             
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
            wins_by_episode = zeros(dataPacketSize-2, this.number_gestures_taken_from_user);
            loses_by_episode = zeros(dataPacketSize-2, this.number_gestures_taken_from_user);
            
            
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
                    
                    [count_wins_in_episode, episode_won, class_mode] = ...
                        this.executeEpisode(num_windows, orientation, dataPacketSize, ...
                                            repTotalTraining, EMG_window_size, Stride, ...
                                            groundTruthIndex, gestureName, ...
                                            verbose_level-1);
                                        
                    wins_by_episode(data_user, episode) = count_wins_in_episode;
                    loses_by_episode(data_user, episode) = num_windows-count_wins_in_episode;
                    
                    % this variable is a reference for the exp replay buffer
                    this.total_num_windows_predicted = this.total_num_windows_predicted + num_windows;

                    
                    % ???????????????????
                    numTestingSamples = max(0.1*num_windows, 20);  % numTestingSamples = max(0.1*numStates, 20);   ????????
                    testStates = zeros(numTestingSamples, 40);
                    % Sampling randomly the data from the experience replay buffer
                    [test_Qval] =  this.sampleRandomFromExpReplay(testStates, episode, momentum, velocity, cost, repTotalTraining);
                    % averageMaxQvalue(episode) = mean( max(test_Qval, [], 2) );
                    this.updateEpsilon(data_user, dataPacketSize-2, episode, this.number_gestures_taken_from_user)
                    
                    
                    
                end
                
            end
            
            weights = reshapeWeights(this.theta, this.qnnOption.numNeuronsLayers);  

            %{
            figure(1);
            subplot(1,3,1);
            plot(1:length(wins_by_episode(1,:)),wins_by_episode(1,:));
            subplot(1,3,2);
            plot(1:length(loses_by_episode(1,:)),loses_by_episode(1,:));
            subplot(1,3,3);
            %}
            accuracy_by_episode = wins_by_episode(1,:)./(wins_by_episode(1,:) + loses_by_episode(1,:));
            
        end
        
        function [count_wins_in_episode, episode_won, class_mode] = executeEpisode(this, num_windows, orientation, dataPacketSize, ...
                                repTotalTraining, EMG_window_size, Stride, ...
                                groundTruthIndex, gestureName, ...
                                verbose_level)
            % -----  El juego del episodio viene a ser el barrido de ventanas en una muestra EMG
            % ----- Se gana el juego si el resultado de reconocimiento es correctom
            % ----- sino se pierde el juego y se tiene una penalizacion
            post_processing      =  evalin('base', 'post_processing');
                    
                    
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
            
            etiquetas_labels_predichas_vector=strings;
            
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
            
            
            
            for window_n=1:num_windows
                
                %Leo datos de muestras
                [~,~,Features_GT,Tiempos_GT,Puntos_GT, ~, ~, ~, groundTruth_GT] = ...
                    Code_1(orientation,dataPacketSize, this.number_gestures_taken_from_user, repTotalTraining, verbose_level-1);

                Vector_EMG_Tiempos_GT(1,window_n)=Tiempos_GT;      %copio primer valor en vector de tiempos de gt
                Vector_EMG_Puntos_GT(1,window_n)=Puntos_GT;        %copio primer valor en vector de tiempos de gt
                
                %---- Defino estado en base a cada ventana EMG
                % window_state is not related with the S or S'
                % because dowsnt matter what action is taken THE NEXT WINDOW STATE WILL BE THE SAME
                state = table2array(Features_GT);
                
                
                
                [Qval, action] = this.selectAction(state);
                
                % AQUI SE VAN GUARDADNO LAS ACCIONES PREDICHAS DENTRO DEl vector de UNA EPOCA
                acciones_predichas_vector(window_n,1)=action;
                % WO    = 1 % WI    = 2 % FIST  = 3 % OPEN  = 4 % PINCH = 5 % RELAX = 6
                etiquetas_labels_predichas_vector(window_n,1)=QNN.convertActionIndexToLabel(action);
                
                real_action=gt_gestures_labels_num(window_n);
                
                [reward, new_state] = this.applyActionAndGetReward(action, real_action, this.Reward_type, state);
                
                if reward == this.amount_reward_correct
                    count_wins_in_episode = count_wins_in_episode + 1;
                end
                
                if verbose_level >=1 && reward~=0
                    fprintf("reward for state %d of %d in actual episode = %d\n", window_n, num_windows, reward);
                end
                
                %---- Experience replay storage ------------------------------------
                idx = mod(this.total_num_windows_predicted + window_n, this.exp_replay_lengthBuffer);
                if idx == 0
                    idx = this.exp_replay_lengthBuffer;
                end
                % the rows of game Replay are 3*minBatchSize
                this.gameReplay(idx, :) = [state, action, reward];   %[state(:)', action, reward, new_state(:)'];
                %---------------------------------------------------------------

                %Acondicionar vectores - si el signo anterior no es igual al signo acual entocnes mido tiempo
                if window_n > 1 && ...
                        etiquetas_labels_predichas_vector(window_n,1) ~= etiquetas_labels_predichas_vector(window_n-1,1)

                    n1=n1+1;
                    ProcessingTimes_vector(1,n1) = toc;  %mido tiempo transcurrido desde ultimo cambio de gesto
                    tic;

                    %obtengo solo etiqueta que se ha venido repetiendo hasta instante numIteration-1
                    etiquetas_labels_predichas_vector_simplif(1,n1)=etiquetas_labels_predichas_vector(window_n-1,1);

                    %obtengo nuevo dato para vector de tiempos
                    TimePoints_vector(1,n1)=Stride*window_n+EMG_window_size/2;           %necesito dato de stride y tamaño de ventana de Victor
                end
                if window_n == num_windows %==maxWindowsAllowed    % si proceso la ultima ventana de la muestra de señal EMG
                    n1=n1+1;
                    ProcessingTimes_vector(1,n1) = toc;  %mido tiempo transcurrido desde ultimo cambio de gesto
                    
                    %obtengo solo etiqueta que no se ha repetido hasta instante numIteration-1
                    etiquetas_labels_predichas_vector_simplif(1,n1)=etiquetas_labels_predichas_vector(window_n,1);

                    %obtengo dato final para vector de tiempos
                    kj=size(groundTruth_GT);  %  se supone q son 1000 puntos
                    TimePoints_vector(1,n1)=  kj(1,2);                 %AQUI CAMBIAR  %1000 puntos

                end
                                
                % state = new_state;
            end
            
            [reward_for_recognition, class_mode] = this.getRewardForRecognition(...
                etiquetas_labels_predichas_vector, etiquetas_labels_predichas_vector_simplif, ...
                post_processing, gestureName, groundTruth_GT, TimePoints_vector, ProcessingTimes_vector);
            
            % boolean, each episode won is a recognition win
            episode_won = reward_for_recognition == this.amount_reward_recognition_correct;
            
            
            if verbose_level >= 1
               fprintf('Eval Recognition: %d\n', reward_for_recognition); 
            end
            
            
        end
        
        function updateEpsilon(this, actual_user, total_users, episode, total_episodes)
            % OENDING: test other epsilon: Annealing the epsilon
            if this.qnnOption.epsilon > 0.10
                this.qnnOption.epsilon = 0.2; %epsilon0*exp(-7*epoch/numEpochs);
            end
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
        
        function [reward, new_state] = applyActionAndGetReward(this, action, real_action, reward_type, state)
            
            if reward_type == 1
                if real_action == action
                    reward = this.amount_reward_correct;
                else
                    reward = this.amount_reward_incorrect;
                end
            else
                reward = 0;
            end
            
            %%%% PENDING: HERE goES ThE fUNCTION FOR CHANGING THE STATE
            % in this first approach is a value in numeric axis x <-|->
            new_state = -1; % state+reward;
            
        end
        
        function [reward_recognition, class_mode] = getRewardForRecognition(this, ...
                etiquetas_labels_predichas_vector, etiquetas_labels_predichas_vector_simplif, ...
                post_processing, gestureName_GT, groundTruth_GT, TimePoints_vector, ProcessingTimes_vector)
            
            
            %=============Saco la moda de los gestos diferentes a NoGesture
            %saco no gesture de este vector para poder usar funcion moda    
            noGestureDetection   = evalin('base', 'noGestureDetection');
            class_mode = QNN.getModeOfGesturesWindowsPredicted(etiquetas_labels_predichas_vector, noGestureDetection);
            
            %----------------------------------------------------
            % POST - Processing: elimino etiquetas espuria usando la
            % moda diferente de no gesto para crear vector de resultados que
            % va a la etaoa de reconocimiento
            

            % GROUND TRUTH (no depende del modelo)------------
            repInfo.gestureName =  gestureName_GT;
            repInfo.groundTruth = groundTruth_GT; %   REV -----

            % PREDICCION--------------------------------------

            if post_processing
                post_processing_result_vector_lables=etiquetas_labels_predichas_vector_simplif;
                dim_vect=size(etiquetas_labels_predichas_vector_simplif);
                for i=1:dim_vect(1,2)
                    if etiquetas_labels_predichas_vector_simplif(1,i) ~= class_mode && etiquetas_labels_predichas_vector_simplif(1,i) ~= "noGesture"
                        post_processing_result_vector_lables(1,i)=class_mode;
                    end  
                end
                response.vectorOfLabels = categorical(post_processing_result_vector_lables); % OK ----- [1,N] % categorical(etiquetas_labels_predichas_vector_simplif); % %CAMBIAR
            else
                response.vectorOfLabels = categorical(etiquetas_labels_predichas_vector_simplif); % OK ----- [1,N] % categorical(etiquetas_labels_predichas_vector_simplif); % %CAMBIAR
            end

            response.vectorOfTimePoints = TimePoints_vector; % OK -----  [40 200 400 600 800 999]; %1xw double  TimePoints_vector                %CAMBIAR
            % tiempo de procesamiento
            response.vectorOfProcessingTimes = ProcessingTimes_vector; % OK -----[0.1 0.1 0.1 0.1 0.1 0.1]; % ProcessingTimes_vector'; % [0.1 0.1 0.1 0.1 0.1 0.1]; % 1xw double                                    %CAMBIAR
            response.class =  categorical(class_mode); % OK ----- categorical({'waveIn'});                %aqui tengo que usar la moda probablemente           %CAMBIAR

            try
                r1 = evalRecognition(repInfo, response);
            catch
                disp('EL vector de predicciones esta compuesto por una misma etiqueta -> Func Eval Recog no funciona');
                r1.recogResult=0;
                if gestureName_GT==response.class
                    r1.classResult=1;
                else
                    r1.classResult=0;
                end

            end
            
            if  r1.recogResult==1
                reward_recognition = this.amount_reward_recognition_correct;
            else
                reward_recognition = this.amount_reward_recognition_incorrect;
            end
        end
        
        function test_Qval = sampleRandomFromExpReplay(this, testStates, episode, momentum, velocity, cost, ...
                repTotalTraining)
            
            
            weights = reshapeWeights(this.theta, this.qnnOption.numNeuronsLayers);  
            options.reluThresh = this.qnnOption.reluThresh;
            options.lambda = this.qnnOption.lambda;
            
            % this wait until de experience_replay is full
            if this.total_num_windows_predicted > ceil( 1.05*this.exp_replay_lengthBuffer )
                
                [~, idx] = sort(rand(this.exp_replay_lengthBuffer, 1));
                randIdx = idx(1:this.qnnOption.miniBatchSize);
                dataX = zeros(this.qnnOption.miniBatchSize, 40);  %64
                dataY = zeros(this.qnnOption.miniBatchSize, 6);
                
                % Computations for the minibatch
                for numExample = 1:this.qnnOption.miniBatchSize
                    % Getting the value of Q(s, a)
                    old_state_er = this.gameReplay(randIdx(numExample), 1:40); %64
                    [~, A] = forwardPropagation(old_state_er, weights,... %old_state_er(:)'
                        this.qnnOption.transferFunctions, options);
                    
                    old_Qval_er = A{end}(:, 2:end);
                    
                    % Getting the value of max_a_Q(s',a')
                    % ????????????????
%                     new_state_er = this.gameReplay(randIdx(numExample), (end - 39):end);  %63
%                     [~, A] = forwardPropagation(new_state_er, weights,... %new_state_er(:)'
%                         transferFunctions, options);
%                     new_Qval_er = A{end}(:, 2:end);
%                     maxQval_er = max(new_Qval_er);

                    maxQval_er = max(old_Qval_er);  %% PENDING: uso el mismo estado como siguiente
                    action_er = this.gameReplay(randIdx(numExample), 41);           %65
                    reward_er = this.gameReplay(randIdx(numExample), 42);           %66
                    
                    if this.Reward_type == 1

                        if reward_er <= -1  %PENDING       OJO CON ESTE
                            % Non-terminal state
                            update_er = reward_er + this.qnnOption.gamma*maxQval_er;
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
                [cost(episode), gradient] = regressionNNCostFunction(dataX, dataY,...
                    this.qnnOption.numNeuronsLayers,...
                    this.theta,...
                    this.qnnOption.transferFunctions,...
                    options);
                % Increase momentum after momIncrease iterations
                % PENDING
                if this.total_num_windows_predicted == this.qnnOption.numEpochsToIncreaseMomentum
                    momentum = options.momentum;
                end
                velocity = momentum*velocity + this.alpha*gradient;
                this.theta = this.theta - velocity;
                % Annealing the learning rate
                this.alpha = this.qnnOption.learningRate*exp(-5*this.total_num_windows_predicted/repTotalTraining);
            end
            [~, A] = forwardPropagation(testStates, weights, this.qnnOption.transferFunctions, options);
            test_Qval = A{end}(:, 2:end);
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
        
        function gesture_mode = getModeOfGesturesWindowsPredicted(etiquetas_labels_predichas_vector, includeNoGesture)
            %GETMODEOFGESTURESPREDICTED

            % dependiendo variable "noGestureDetection" Saco moda 1) sin considerar no gesto ,o 2) considerando no gesto

            if includeNoGesture
                gesture_mode=mode(categorical(etiquetas_labels_predichas_vector));    %Saco moda incluyendo etiqueta de NoGesture
            else
                etiquetas_labels_predichas_vector_without_NoGesture=strings;
                %-------- REVISAR-ESTO HABILITAR SI REQUIERO QUITAR NO GESTURE -----
                for i=1:length(etiquetas_labels_predichas_vector)
                    if etiquetas_labels_predichas_vector(i,1)~="noGesture"
                        etiquetas_labels_predichas_vector_without_NoGesture(i,1)=etiquetas_labels_predichas_vector(i,1);
                    end
                end
                gesture_mode=mode(categorical(etiquetas_labels_predichas_vector_without_NoGesture));  %Saco la moda de las etiquetas dif a no gesture
            end

            %Si al sacar la moda todo es no gesto (<missing>), entonces la moda es no gesto
            if ismissing(gesture_mode)
                gesture_mode="noGesture";
            end


        end
    end
end

