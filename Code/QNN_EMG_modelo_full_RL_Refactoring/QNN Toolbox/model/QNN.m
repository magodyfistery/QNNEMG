classdef QNN < handle
    
    properties(Constant)
       gesture_names = ["waveOut", "waveIn", "fist", "open", "pinch", "noGesture"]; 
    end
    
    properties
        qnnOption;
        Reward_type;  % on si quiero recompensa -1 x ventana (clasif) y -10 x recog
        amount_reward_correct = 1;
        amount_reward_incorrect = -1;
        amount_reward_recognition_correct = 10;
        amount_reward_recognition_incorrect = -10;
        
        number_gestures_taken_from_user;  % This is RepTraining, the same
        theta;
        theta_freeze;
        exp_replay_lengthBuffer;
        % gameReplay -> exp replay with dims: exp_replay_lengthBufferx4
        % 4 = 1 reward, 1 action, 40 of length s, 0 of length of new s'
        % each state is a numeric value is scalar
        gameReplay;
        total_num_windows_predicted = 0;
        known = 0;
        dont_known = 0;
        
        num_correct_predictions = 0;
        training_cost = [];
        
        % auxiliar
        index_gesture;
        reserved_space_for_gesture; % reserved space for gesture in game replay
        
        alpha;
        velocity;
        repTotalTraining;
        
        cost = [];
        total_episodes = 0;
        update_count = 0;
        
    end
    
    methods
        function obj = QNN(qnnOption, Reward_type, number_gestures_taken_from_user, reserved_space_for_gesture)
            %QNN Construct an instance of this class
            obj.qnnOption = qnnOption;
            
            obj.Reward_type = Reward_type;
            obj.number_gestures_taken_from_user = number_gestures_taken_from_user;
            % Experience replay initialization ------------------------------
            obj.reserved_space_for_gesture = reserved_space_for_gesture; % 3 * ceil(qnnOption.miniBatchSize/6);
            obj.exp_replay_lengthBuffer = numel(QNN.gesture_names) * obj.reserved_space_for_gesture;
            s  =  nan(obj.exp_replay_lengthBuffer, 40);  % 1 for optimization
            sp =  nan(obj.exp_replay_lengthBuffer, 40);
            a  =  nan(obj.exp_replay_lengthBuffer, 1);
            r  =  nan(obj.exp_replay_lengthBuffer, 1);
            obj.gameReplay = [s, a, r, sp]; %(state, action, reward, state') , sp
            
            obj.index_gesture = zeros(1, numel(QNN.gesture_names));
            
        end
        
        function initTheta(this, theta)
            this.theta = theta;
            this.theta_freeze = this.theta(:);
        end
        
        function [summary_episodes, summary_classifications_mode, ... 
                wins_by_episode, loses_by_episode] = train(this, verbose_level)
            
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
            this.repTotalTraining = repTotalTraining;
            
            this.alpha = this.qnnOption.learningRate;
            this.velocity = zeros( size(this.theta) );
            
            
            wins_episodes = zeros(1, dataPacketSize-2);
            losses_episodes = zeros(1, dataPacketSize-2);
            
            wins_clasification_with_mode = zeros(1, dataPacketSize-2);
            losses_clasification_with_mode = zeros(1, dataPacketSize-2);
            
            wins_by_episode = zeros(dataPacketSize-2, this.number_gestures_taken_from_user);
            loses_by_episode = zeros(dataPacketSize-2, this.number_gestures_taken_from_user);
            
            
            debug_step = ceil(this.number_gestures_taken_from_user/2); %%% Just for debugging
            
            dataPacket = evalin('base','dataPacket');
            
                
            % for each user in Specific
            for data_user=1:dataPacketSize-2
                
                 %%%% PENDING
                % num_windows = 1 + floor((1000-EMG_window_size)/Stride);  % num_windows = NumeroVentanasGT
                % numTestingSamples = max(0.1*(num_windows+1), 20);  % numTestingSamples = max(0.1*num_windows, 20);   ????????
                % testStates = zeros(numTestingSamples, 40);  % ???????????????????????????????????????
                % etiquetas_labels_predichas_matrix=strings(num_windows,numEpochs);
            
                assignin('base', 'userIndex', data_user+2);
                userData = loadSpecificUser(dataPacket, data_user+2);
                energy_index = strcmp(orientation(:,1), userData.userInfo.name);
                rand_data=orientation{energy_index,6};
                
                
                
                % Each episode is a gesture taken from user
                % each episode has num_windows with the window_size/stride
                for episode=1:this.number_gestures_taken_from_user
                    
                    if (mod(episode, debug_step) == 0 || episode == 1 || episode == this.number_gestures_taken_from_user)  && verbose_level >= 1
                        fprintf("TRAINING| User: %s (%d of %d), Episode(Gesture) %d of %d\n", userData.userInfo.name, data_user, dataPacketSize-2, episode, this.number_gestures_taken_from_user);
                    end
                   
                    emgRepetition = evalin('base','emgRepetition');
                    numberPointsEmgData = length(userData.training{rand_data(emgRepetition),1}.emg);
                    num_windows = getNumberWindows(numberPointsEmgData, EMG_window_size, Stride, false);
                    groundTruthIndex = userData.training{rand_data(emgRepetition),1}.groundTruthIndex;
                    gestureName = userData.training{rand_data(emgRepetition),1}.gestureName;
                    
                    [count_wins_in_episode, episode_won, class_mode] = ...
                        this.executeEpisode(num_windows, orientation, dataPacketSize, ...
                                            this.number_gestures_taken_from_user, repTotalTraining, EMG_window_size, Stride, ...
                                            groundTruthIndex, gestureName, ...
                                            verbose_level-1, false);
                                        
                    wins_by_episode(data_user, episode) = count_wins_in_episode;
                    loses_by_episode(data_user, episode) = num_windows-count_wins_in_episode;
                    
                    wins_clasification_with_mode(1, data_user) = wins_clasification_with_mode(1, data_user) + (class_mode == gestureName);
                    losses_clasification_with_mode(1, data_user) = losses_clasification_with_mode(1, data_user) + (class_mode ~= gestureName);
                    
                    wins_episodes(1, data_user) = wins_episodes(1, data_user) + episode_won;
                    losses_episodes(1, data_user) = losses_episodes(1, data_user) + ~episode_won;
                    
                    % this.updateEpsilon(data_user, dataPacketSize-2, episode, this.number_gestures_taken_from_user)
                    
                    this.total_episodes = this.total_episodes + 1;
                    
                    this.theta_freeze = this.theta(:);
                    
                    
                    
                    
                end
                
            end
            
            % END TRAINING
            summary_episodes = [wins_episodes; losses_episodes];
            summary_classifications_mode = [wins_clasification_with_mode; losses_clasification_with_mode];
            
            
        end
        
        function [count_wins_in_episode, episode_won, class_mode] = executeEpisode(this, num_windows, orientation, dataPacketSize, ...
                                number_gestures_taken_from_user, repTotalTraining, EMG_window_size, Stride, ...
                                groundTruthIndex, gestureName, ...
                                verbose_level, is_test)
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
            
            %Leo datos de muestras
            window_n = 1;
            [~,~,Features_GT,Tiempos_GT,Puntos_GT, ~, ~, ~, groundTruth_GT] = ...
                Code_1(orientation,dataPacketSize, number_gestures_taken_from_user, repTotalTraining, verbose_level-1);

            Vector_EMG_Tiempos_GT(1,window_n)=Tiempos_GT;      %copio primer valor en vector de tiempos de gt
            Vector_EMG_Puntos_GT(1,window_n)=Puntos_GT;        %copio primer valor en vector de tiempos de gt

            %---- Defino estado en base a cada ventana EMG
            % window_state is not related with the S or S'
            % because dowsnt matter what action is taken THE NEXT WINDOW STATE WILL BE THE SAME
            state = table2array(Features_GT);

            
            while window_n < num_windows-1
                
                [~, action] = this.selectAction(state, is_test);
                
                % AQUI SE VAN GUARDADNO LAS ACCIONES PREDICHAS DENTRO DEl vector de UNA EPOCA
                acciones_predichas_vector(window_n,1)=action;
                % WO    = 1 % WI    = 2 % FIST  = 3 % OPEN  = 4 % PINCH = 5 % RELAX = 6
                etiquetas_labels_predichas_vector(window_n,1)=QNN.convertActionIndexToLabel(action);
                
                real_action=gt_gestures_labels_num(window_n+1);
                
                [reward, ~] = this.applyActionAndGetReward(action, real_action, this.Reward_type, state);
                
                if reward == this.amount_reward_correct
                    count_wins_in_episode = count_wins_in_episode + 1;
                end
                
                if verbose_level >=1 && reward~=0
                    fprintf("reward for state %d of %d in actual episode = %d\n", window_n, num_windows, reward);
                end
                
                [~,~,Features_GT,Tiempos_GT,Puntos_GT, ~, ~, ~, groundTruth_GT] = ...
                    Code_1(orientation,dataPacketSize, number_gestures_taken_from_user, repTotalTraining, verbose_level-1);

                Vector_EMG_Tiempos_GT(1,window_n+1)=Tiempos_GT;      %copio primer valor en vector de tiempos de gt
                Vector_EMG_Puntos_GT(1,window_n+1)=Puntos_GT;        %copio primer valor en vector de tiempos de gt

                %---- Defino estado en base a cada ventana EMG
                % window_state is not related with the S or S'
                % because dowsnt matter what action is taken THE NEXT WINDOW STATE WILL BE THE SAME
                new_state = table2array(Features_GT);
                
                if ~is_test
                    this.total_num_windows_predicted = this.total_num_windows_predicted + 1;
                    this.saveExperienceReplay(state, action, reward, new_state);
                    
                    this.learnFromExperienceReplay();
                end
                
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
                
                state = new_state;
                window_n = window_n + 1;
                
                
                
                                
            end
            
            [reward_for_recognition, class_mode] = this.getRewardForRecognition(...
                etiquetas_labels_predichas_vector, etiquetas_labels_predichas_vector_simplif, ...
                post_processing, gestureName, groundTruth_GT, TimePoints_vector, ProcessingTimes_vector, verbose_level-1);
            
            % boolean, each episode won is a recognition win
            episode_won = reward_for_recognition == this.amount_reward_recognition_correct;
            
            
            if verbose_level >= 1
               fprintf('Eval Recognition: %d\n', reward_for_recognition); 
            end
            
            
        end
        
        function updateEpsilon(this, ~, ~, ~, ~)
            % OENDING: test other epsilon: Annealing the epsilon
            if this.qnnOption.epsilon > 0.10
                this.qnnOption.epsilon = 0.1; %epsilon0*exp(-7*epoch/numEpochs);
            end
        end
        
        function [Qval, action_index] = selectAction(this, state, is_test)
            
            options.reluThresh = this.qnnOption.reluThresh;
            options.lambda = this.qnnOption.lambda;
            weights = reshapeWeights(this.theta, this.qnnOption.numNeuronsLayers); 
            [dummyVar, A] = forwardPropagation(state, weights, this.qnnOption.transferFunctions, options); % 6 Valores de Q - uno por cada accion
            Qval = A{end}(:, 2:end);  
            [~, idx] = max(Qval); % obtengo indice de Qmax a partir de vector Qval

            if ~is_test
                % Epsilon-greedy action selection - REV Con epsilon=1
                % Inicialmente hace solo exploracion, luego el valor de epsilon se va reduciendo a medida que tengo mas informacion
                % Si rand <= epsilon, obtengo un Q de manera aleatoria, el cual será diferente a Qmax (exploracion)

            
                if rand <= this.qnnOption.epsilon            % siempre se cumple con epsilon=1% %Initial value of epsilon for the epsilon-greedy exploration
                    full_action_list = 1:6;    %actionList = 1:6;               % posibles acciones  (AQUI HAY QUE CAMBIAR - #clases) usar gestos
                    actionList = full_action_list(full_action_list ~= idx);           %  Crea lista con las acciones q no tienen Qmax
                    idx_valid_action = randi([1 length(actionList)]);
                    idx = full_action_list(actionList(idx_valid_action));
                end
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
            % in this problem of MDP its not necesary a new state
            % in this first approach is a value in numeric axis x <-|->
            new_state = state; % state+reward;
            
        end
        
        function [reward_recognition, class_mode] = getRewardForRecognition(this, ...
                etiquetas_labels_predichas_vector, etiquetas_labels_predichas_vector_simplif, ...
                post_processing, gestureName_GT, groundTruth_GT, TimePoints_vector, ProcessingTimes_vector, verbose_level)
            
            
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
                %PENDING DEPURAR PORQUE PASA ESTO
                if verbose_level >= 1
                    disp('EL vector de predicciones esta compuesto por una misma etiqueta -> Func Eval Recog no funciona');
                end
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
        
        
        function saveExperienceReplay(this, state, action, reward, new_state)
                    
            %---- Experience replay storage ------------------------------------
            
            % if reward < 0
                % this.num_correct_predictions = this.num_correct_predictions + 1;
                
            this.index_gesture(action) = this.index_gesture(action) + 1;
            index_experience_replay = mod(this.index_gesture(action), this.reserved_space_for_gesture);
            if index_experience_replay == 0
                index_experience_replay = this.reserved_space_for_gesture;
            end

            offset = (action-1) * this.reserved_space_for_gesture;
            this.gameReplay(offset+index_experience_replay, :) = [state, action, reward, new_state];   %[state(:)', action, reward, new_state(:)'];

            
        end
            
        function learnFromExperienceReplay(this)
            
            options.reluThresh = this.qnnOption.reluThresh;
            options.lambda = this.qnnOption.lambda;
            
            weights = reshapeWeights(this.theta, this.qnnOption.numNeuronsLayers); 
            weights_freeze = reshapeWeights(this.theta_freeze, this.qnnOption.numNeuronsLayers); 
            
            valid_replay = getRowsNotNan(this.gameReplay);
            amount_data_not_NAN =  size(valid_replay, 1); % this.exp_replay_lengthBuffer;
            
            if amount_data_not_NAN < this.qnnOption.miniBatchSize
                return;
            end
            
            [~, idx] = sort(rand(this.qnnOption.miniBatchSize, 1));
            randIdx = idx(1:this.qnnOption.miniBatchSize);
            

            dataX = zeros(this.qnnOption.miniBatchSize, 40);  %64
            dataY = zeros(this.qnnOption.miniBatchSize, 6);
            % Computations for the minibatch
            for numExample=1:this.qnnOption.miniBatchSize

                % Getting the value of Q(s, a)
                old_state_er = valid_replay(randIdx(numExample), 1:40); %64
                [dummyVar, A] = forwardPropagation(old_state_er, weights,... %old_state_er(:)'
                    this.qnnOption.transferFunctions, options);
                old_Qval_er = A{end}(:, 2:end);
                % Getting the value of max_a_Q(s',a')
                new_state_er = valid_replay(randIdx(numExample), (end - 39):end);  %63
                [dummyVar, A] = forwardPropagation(new_state_er, weights_freeze,... %new_state_er(:)'
                    this.qnnOption.transferFunctions, options);
                new_Qval_er = A{end}(:, 2:end);
                maxQval_er = max(new_Qval_er);
                action_er = valid_replay(randIdx(numExample), 41);           %65
                reward_er = valid_replay(randIdx(numExample), 42);

                
                % Data for training the ANN
                dataX(numExample, :) = old_state_er;  %old_state_er(:)'
                dataY(numExample, :) = old_Qval_er;
                
                dataY(numExample, action_er) = reward_er + this.qnnOption.gamma*maxQval_er;
                
                
            end
            this.update_count = this.update_count + 1;
            % Updating the weights of the ANN
            [this.cost(this.update_count), gradient] = ...
                regressionNNCostFunction(dataX, dataY,...
                this.qnnOption.numNeuronsLayers,...
                this.theta,...
                this.qnnOption.transferFunctions,...
                options);
            % Increase momentum after momIncrease iterations
            if this.total_episodes == this.qnnOption.numEpochsToIncreaseMomentum
                this.qnnOption.initialMomentum = this.qnnOption.momentum;
            end
            this.velocity = this.qnnOption.initialMomentum*this.velocity + this.alpha*gradient;
            this.theta = this.theta - this.velocity;
                    
            % Annealing the learning rate
            this.alpha = this.qnnOption.learningRate*exp(-5*this.total_episodes/this.repTotalTraining);
            
        end

        function [summary_episodes, summary_classifications_mode, ... 
                wins_by_episode, loses_by_episode] = test(this, verbose_level, num_gestures_validation)
            
            EMG_window_size = evalin('base', 'WindowsSize');                                                %AQUI PONER WINDOW SIZE
            Stride = evalin('base', 'Stride');
            orientation      = evalin('base', 'orientation');
            dataPacketSize     = evalin('base', 'dataPacketSize');
            repTotalTraining  = num_gestures_validation*(dataPacketSize-2);  % is the same of repTotalTraining
            
             
            wins_episodes = zeros(1, dataPacketSize-2);
            losses_episodes = zeros(1, dataPacketSize-2);
            
            wins_clasification_with_mode = zeros(1, dataPacketSize-2);
            losses_clasification_with_mode = zeros(1, dataPacketSize-2);
            
            
            
            
            wins_by_episode = zeros(dataPacketSize-2, num_gestures_validation);
            loses_by_episode = zeros(dataPacketSize-2, num_gestures_validation);
            
            
            debug_step = ceil(num_gestures_validation/5); %%% Just for debugging
            dataPacket = evalin('base','dataPacket');
            % for each user in Specific
            for data_user=1:dataPacketSize-2
                
                assignin('base', 'userIndex', data_user+2);
                userData = loadSpecificUser(dataPacket, data_user+2);
                energy_index = strcmp(orientation(:,1), userData.userInfo.name);
                rand_data=orientation{energy_index,6};
                
                for episode=1:num_gestures_validation
                    
                    if (mod(episode, debug_step) == 0 || episode == 1 || episode == num_gestures_validation) && verbose_level >= 1
                        fprintf("TESTING| User: %s (%d of %d), Episode(Gesture) %d of %d\n", userData.userInfo.name, data_user, dataPacketSize-2,  episode, num_gestures_validation);
                    end
                   
                    
                    emgRepetition = evalin('base','emgRepetition');
                    numberPointsEmgData = length(userData.training{rand_data(emgRepetition),1}.emg);
                    num_windows = getNumberWindows(numberPointsEmgData, EMG_window_size, Stride, false);
                    groundTruthIndex = userData.training{rand_data(emgRepetition),1}.groundTruthIndex;
                    gestureName = userData.training{rand_data(emgRepetition),1}.gestureName;
                    
                    [count_wins_in_episode, episode_won, class_mode] = ...
                        this.executeEpisode(num_windows, orientation, dataPacketSize, ...
                                            num_gestures_validation, repTotalTraining, EMG_window_size, Stride, ...
                                            groundTruthIndex, gestureName, ...
                                            verbose_level-1, true);
                                        
                                        
                                        
                    wins_by_episode(data_user, episode) = count_wins_in_episode;
                    loses_by_episode(data_user, episode) = num_windows-count_wins_in_episode;
                    wins_clasification_with_mode(1, data_user) = wins_clasification_with_mode(1, data_user) + (class_mode == gestureName);
                    losses_clasification_with_mode(1, data_user) = losses_clasification_with_mode(1, data_user) + (class_mode ~= gestureName);
                    wins_episodes(1, data_user) = wins_episodes(1, data_user) + episode_won;
                    losses_episodes(1, data_user) = losses_episodes(1, data_user) + ~episode_won;                   
                    
                end
                
            end
            
            % END TESTING
            summary_episodes = [wins_episodes; losses_episodes];
            summary_classifications_mode = [wins_clasification_with_mode; losses_clasification_with_mode];
            
        end

        
    end
    
    methods(Static)
        function action_text = convertActionIndexToLabel(action)
            action_text = QNN.gesture_names(action);                
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

