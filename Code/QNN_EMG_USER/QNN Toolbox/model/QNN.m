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
            % sp =  nan(obj.exp_replay_lengthBuffer, 1);
            a  =  nan(obj.exp_replay_lengthBuffer, 1);
            r  =  nan(obj.exp_replay_lengthBuffer, 1);
            obj.gameReplay = [s, a, r]; %(state, action, reward, state') , sp
            
            obj.index_gesture = zeros(1, numel(QNN.gesture_names));
            
        end
        
        function initTheta(this, theta)
            this.theta = theta;
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
                    
                    if mod(episode, debug_step) == 0 && verbose_level >= 1 || episode == 1 || episode == this.number_gestures_taken_from_user
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
            
            
            
            for window_n=1:num_windows
                
                %Leo datos de muestras
                [~,~,Features_GT,Tiempos_GT,Puntos_GT, ~, ~, ~, groundTruth_GT] = ...
                    Code_1(orientation,dataPacketSize, number_gestures_taken_from_user, repTotalTraining, verbose_level-1);

                Vector_EMG_Tiempos_GT(1,window_n)=Tiempos_GT;      %copio primer valor en vector de tiempos de gt
                Vector_EMG_Puntos_GT(1,window_n)=Puntos_GT;        %copio primer valor en vector de tiempos de gt
                
                %---- Defino estado en base a cada ventana EMG
                % window_state is not related with the S or S'
                % because dowsnt matter what action is taken THE NEXT WINDOW STATE WILL BE THE SAME
                state = table2array(Features_GT);
                
                
                
                [~, action] = this.selectAction(state, is_test);
                
                % AQUI SE VAN GUARDADNO LAS ACCIONES PREDICHAS DENTRO DEl vector de UNA EPOCA
                acciones_predichas_vector(window_n,1)=action;
                % WO    = 1 % WI    = 2 % FIST  = 3 % OPEN  = 4 % PINCH = 5 % RELAX = 6
                etiquetas_labels_predichas_vector(window_n,1)=QNN.convertActionIndexToLabel(action);
                
                real_action=gt_gestures_labels_num(window_n);
                
                [reward, ~] = this.applyActionAndGetReward(action, real_action, this.Reward_type, state);
                
                if reward == this.amount_reward_correct
                    count_wins_in_episode = count_wins_in_episode + 1;
                end
                
                if verbose_level >=1 && reward~=0
                    fprintf("reward for state %d of %d in actual episode = %d\n", window_n, num_windows, reward);
                end
                
                if ~is_test
                    this.total_num_windows_predicted = this.total_num_windows_predicted + 1;
                    this.updateWeigthsWithExperienceReplay(state, action, reward);
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
        
        function updateEpsilon(this, ~, ~, ~, ~)
            % OENDING: test other epsilon: Annealing the epsilon
            if this.qnnOption.epsilon > 0.10
                this.qnnOption.epsilon = 0.1; %epsilon0*exp(-7*epoch/numEpochs);
            end
        end
        
        function [Qval, action_index] = selectAction(this, state, is_test)
            
            Qval = this.feedForward(state); % 6 Valores de Q - uno por cada accion

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
                %PENDING DEPURAR PORQUE PASA ESTO
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
        
        
        function updateWeigthsWithExperienceReplay(this, state, action, reward)
                    
            %---- Experience replay storage ------------------------------------
            
            % if reward < 0
                % this.num_correct_predictions = this.num_correct_predictions + 1;
                
                this.index_gesture(action) = this.index_gesture(action) + 1;
                index_experience_replay = mod(this.index_gesture(action), this.reserved_space_for_gesture);
                if index_experience_replay == 0
                    index_experience_replay = this.reserved_space_for_gesture;
                end
                
                offset = (action-1) * this.reserved_space_for_gesture;
                this.gameReplay(offset+index_experience_replay, :) = [state, action, reward];   %[state(:)', action, reward, new_state(:)'];

            % end
            
            this.learnFromExperienceReplay();
        end
            
        function learnFromExperienceReplay(this)
            
            
            valid_replay = getRowsNotNan(this.gameReplay);
            amount_data_not_NAN =  size(valid_replay, 1); % this.exp_replay_lengthBuffer;
            
            actual_batch_size = min(this.qnnOption.miniBatchSize, amount_data_not_NAN);
            
            [~, idx] = sort(rand(amount_data_not_NAN, 1));
            randIdx = idx(1:actual_batch_size);
            

            % Computations for the minibatch
            for numExample=1:actual_batch_size

                % Getting the value of Q(s, a)
                
                experience_replay_state = valid_replay(randIdx(numExample), 1:40);
                experience_replay_action = valid_replay(randIdx(numExample), 41);
                experience_replay_reward = valid_replay(randIdx(numExample), 42);

                q_sp_a = experience_replay_action;

                if experience_replay_reward > 0
                    % Getting the value of Q(s', a)
                    % i know the prediction was correct
                    is_custom_sparsed = false;
                    this.known = this.known + 1;
                else
                    % i dont know what gesture is
                    is_custom_sparsed = true;
                    this.dont_known = this.dont_known + 1;
                end

                % Data for training the ANN
                % Updating the weights of the ANN
                [gradient, cost] = this.calculateGradientForOneObservation(experience_replay_state, q_sp_a, is_custom_sparsed);
                this.theta = this.theta - (this.qnnOption.learningRate * gradient);
                
                % taking cost of each update
                this.training_cost = [this.training_cost cost];
                

            end
            
        end
        
        function output_nn = feedForward(this, X)
            % returns the output of neural network
            
            m = size(X, 1);
            number_layers = size(this.qnnOption.numNeuronsLayers, 2);
            index_reshape_begin = 1;
            index_reshape_end = 0;
            a = [ones(m, 1) X];
            
            for i=1:number_layers-1
                index_reshape_end = index_reshape_end + (this.qnnOption.numNeuronsLayers(i)+1)*this.qnnOption.numNeuronsLayers(i+1);
                weights = reshape(this.theta(index_reshape_begin:index_reshape_end), (this.qnnOption.numNeuronsLayers(i)+1), this.qnnOption.numNeuronsLayers(i+1));

                z = a * weights;
                switch(this.qnnOption.transferFunctions{i+1})
                    case "sigmoid"
                        a = [ones(m, 1) sigmoid(z)];
                    otherwise
                        disp("Error, the transfer function is not supported");
                        a = [ones(m, 1) sigmoid(z)];
                end
                index_reshape_begin = index_reshape_end + 1;
            end
            
            output_nn = a(:, 2:end);
        end

        function [gradient, cost] = calculateGradientForOneObservation(this, X, y, is_custom_sparse_y)

            % X es solo un ejemplo
            m = size(X, 1);
            number_layers = size(this.qnnOption.numNeuronsLayers, 2);
            
            %%% Feed Forward
            index_reshape_begin = 1;
            index_reshape_end = 0;
            indexes_reshapes_theta = zeros(number_layers-1, 2);

            a = [ones(m,1) X];
            activations_functions = zeros(sum(this.qnnOption.numNeuronsLayers)+number_layers-1, 1);
            activations_derivates_functions = zeros(sum(this.qnnOption.numNeuronsLayers)+number_layers-1, 1);

            indexes_activations_functions = zeros(number_layers, 2);  % 2 por ser inicio y fin
            index_activations_begin = 1;
            index_activations_end = this.qnnOption.numNeuronsLayers(1)+1;

            activations_functions(index_activations_begin:index_activations_end, 1) = a(:);
            indexes_activations_functions(1, :) = [index_activations_begin index_activations_end];

            index_activations_begin = index_activations_end+1;



            for i=1:number_layers-1
                index_reshape_end = index_reshape_end + (this.qnnOption.numNeuronsLayers(i)+1)*this.qnnOption.numNeuronsLayers(i+1);
                weights = reshape(this.theta(index_reshape_begin:index_reshape_end), (this.qnnOption.numNeuronsLayers(i)+1), this.qnnOption.numNeuronsLayers(i+1));


                z = a * weights;
                switch(this.qnnOption.transferFunctions{i+1})
                    case "sigmoid"
                        a = [ones(1,1) sigmoid(z)];
                        a_derivate = [ones(1,1) sigmoidGradient(z)];
                    otherwise
                        disp("Error, the transfer function is not supported");
                        a = [ones(1,1) sigmoid(z)];
                        a_derivate = [ones(1,1) sigmoidGradient(z)];
                end

                index_activations_end = index_activations_end + this.qnnOption.numNeuronsLayers(i+1)+1;

                activations_functions(index_activations_begin:index_activations_end, 1) = a(:);
                activations_derivates_functions(index_activations_begin:index_activations_end, 1) = a_derivate(:);

                indexes_reshapes_theta(i, :) = [index_reshape_begin index_reshape_end];
                indexes_activations_functions(i+1, :) = [index_activations_begin index_activations_end];



                index_reshape_begin = index_reshape_end + 1;
                index_activations_begin = index_activations_end+1;
            end

            gradient = zeros(size(this.theta));

            
            % Backpropagation
            ind = indexes_activations_functions(number_layers, :);
            % la ultima capa de activación no requiere el uno agregado del bias
            h_t = activations_functions(ind(1):ind(2));
            h = h_t(2:end)';  % ignora el bias
            
            
            % hipotesis es a, la respuesta correcta es y (hot encoded)
            if is_custom_sparse_y
                % sparse_y = a(2:end);
                % sparse_y(y) = 0;
                % INVERSE sparse. [1 1 1 0 1 1]
                sparse_y = sparse_one_hot_encoding(-y, this.qnnOption.numNeuronsLayers(number_layers));
            else
                sparse_y = sparse_one_hot_encoding(y, this.qnnOption.numNeuronsLayers(number_layers));
            end

            % sum(m x num_labels) => 1 x num_labels; sum(1 x num_labels) => 1x1
            % PENDING regulariztion
            regularization = 0; % (lambda/(2*m)) * (sum(sum( Theta1(:, 2:end) .^ 2 )) + sum(sum( Theta2(:, 2:end) .^ 2 )));

            cost = sum(sum( -sparse_y .* log(a(2:end)) - (1 - sparse_y) .* log(1 - a(2:end))))/m + regularization;
            
%             if cost > 4
%                disp("Why?"); 
%             end
                
            delta = h - sparse_y;
            
            
            
            
            

            i=number_layers-1;

            while i >= 1
                ind = indexes_activations_functions(i, :);
                ind_theta = indexes_reshapes_theta(i, :);
                a = activations_functions(ind(1):ind(2))';  % se requiere el bias
                a_d = activations_derivates_functions(ind(1):ind(2));
                a_derivate = a_d(2:end)';
                theta_t = reshape(this.theta(ind_theta(1):ind_theta(2)), this.qnnOption.numNeuronsLayers(i)+1, this.qnnOption.numNeuronsLayers(i+1));

                weights = theta_t(2:end,:)';
                
                % delta
                grad = zeros(size(weights, 1), size(weights, 2) + 1);
                grad = (grad + delta' * a)';

                gradient(ind_theta(1):ind_theta(2), 1) = grad(:);
                if i > 1
                  delta = (delta * weights) .* a_derivate;
                end
                i = i - 1;

            end
            
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
                    
                    if mod(episode, debug_step) == 0 && verbose_level >= 1 || episode == 1 || episode == num_gestures_validation
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

