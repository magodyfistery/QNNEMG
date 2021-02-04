function result = testQNN_Exp_Replay(weights, transferFunctions, options, typeWorld, typeControl)
% Creating a new instance of the world
%state = createWorld(typeWorld);

%Leo ventana inicial de cierta muestra --------------------------------------------
[Numero_Ventanas_GT,EMG_GT,Features_GT,Tiempos_GT,Puntos_GT,Usuario_GT,gestureName_GT,groundTruthIndex_GT,groundTruth_GT] = ...
    Code_1(orientation,dataPacketSize,RepTraining,RepTotalTraining);

while Usuario_GT=="NaN"
    
    if Usuario_GT~="NaN"
        break
    end
    
    [Numero_Ventanas_GT,EMG_GT,Features_GT,Tiempos_GT,Puntos_GT,Usuario_GT,gestureName_GT,groundTruthIndex_GT,groundTruth_GT] = ...
        Code_1(orientation,dataPacketSize,RepTraining,RepTotalTraining);
end
%-----------------------------------------------------------------------------------
window_n=window_n+1;                               %window_n == 1
Vector_EMG_Tiempos_GT=zeros(1,Numero_Ventanas_GT); %creo vector de tiempos de gt
Vector_EMG_Puntos_GT=zeros(1,Numero_Ventanas_GT);  %creo vector de puntos de gt
Vector_EMG_Tiempos_GT(1,window_n)=Tiempos_GT;      %copio primer valor en vector de tiempos de gt
Vector_EMG_Puntos_GT(1,window_n)=Puntos_GT;        %copio primer valor en vector de tiempos de gt

%---- Defino estado inicial en base a cada ventana EMG  -----------------------------------------------------*
%state = rand(1, 40);
state =table2array(Features_GT);           %Defino ESTADO inicial
disp('initial state')
%disp(state)
%disp('state')
%disp(size(state))
Reward_type      =  evalin('base', 'Reward_type');
%Each_complete_signal = rand(1, 40*Numero_Ventanas_GT); %rand(1, 40*maxWindowsAllowed);      %AQUI CAMBIAR, HAY QUE PONER CADA SEñAL COMPLETA AQUI  %&&&&&&&&&&&&&&&&

% ---- Inicializo variables requeridas para guardar datos de prediccion
etiquetas = 1+round((5)*rand(Numero_Ventanas_GT,1));   %1+round((5)*rand(maxWindowsAllowed,1)); %%%%%%%% AQUI PONER ground truth de cada ventana EMG - gestos de 1 a 6
etiquetas_labels_predichas_vector=strings;
%etiquetas_labels_predichas_matrix=strings;
etiquetas_labels_predichas_vector_without_NoGesture=strings;
%etiquetas_labels_predichas_matrix_without_NoGesture=strings;
acciones_predichas_vector = zeros(Numero_Ventanas_GT,1);%zeros(maxWindowsAllowed,1);         %%%%%%%%   EN ESTE VECTOR VAN A IR LAS ACCIONES PREDICHAS, LAS

% ---- inicializo parametros medicion tiempo, y vectores de prediccion
% necesarios para evaluar reconocimiento en cada epoca -----------------
ProcessingTimes_vector=[];
TimePoints_vector=[];
n1=0;
etiquetas_labels_predichas_vector_simplif=strings;
%----------------------------------------------------------------

% if strcmp(typeControl, 'AI')
%     displayWorld(state);
% else
%     displayWorld(state, false);
% end

gameOn = true;
maxNumSteps = 10;
stepNum = 0;

while gameOn
    
    numIteration = numIteration + 1;
    
    stepNum = stepNum + 1;
    %if strcmp(typeControl, 'AI')
    [dummyVar, A] = forwardPropagation(state(:)', weights,...
        transferFunctions, options);
    Qval = A{end}(:, 2:end);
    [dummyVar, action] = max(Qval);
    
    acciones_predichas_vector(numIteration,1)=action;   % AQUI SE VAN GUARDADNO LAS ACCIONES PREDICHAS DENTRO DEl vector de UNA EPOCA
    
    %     else
    %         % Read the action from keyboard
    %         keyPressed = getkey();
    %         % up = 30, down = 31, left = 28, right = 29
    %         if keyPressed == 30, action = 1; end
    %         if keyPressed == 31, action = 2; end
    %         if keyPressed == 28, action = 3; end
    %         if keyPressed == 29, action = 4; end
    %     end
    % Taking the selected action
    %     if strcmp(typeControl, 'AI')
    %         displayAction(state, action);
    %     else
    %         displayAction(state, action, false);
    %     end
    
    [Numero_Ventanas_GT,EMG_GT,Features_GT,Tiempos_GT,Puntos_GT,Usuario_GT,gestureName_GT,groundTruthIndex_GT,groundTruth_GT] = ...
        Code_1(orientation,dataPacketSize,RepTraining,RepTotalTraining);
    
    while Usuario_GT=="NaN"
        if Usuario_GT~="NaN"
            break
        end
        [Numero_Ventanas_GT,EMG_GT,Features_GT,Tiempos_GT,Puntos_GT,Usuario_GT,gestureName_GT,groundTruthIndex_GT,groundTruth_GT] = ...
            Code_1(orientation,dataPacketSize,RepTraining,RepTotalTraining);
    end
    
    window_n=window_n+1; %window_n == 2 hasta window final
    
    %---- Vectorizo datos necesarios de vectores de timpos y de puntos de Ground truth -----------------*
    %  if window_n==Numero_Ventanas_GT -> FIN DE JUEGO
    if window_n==Numero_Ventanas_GT%+1  %REV este +1  %window_n==Numero_Ventanas_GT
        Vector_EMG_Tiempos_GT(1,window_n)=Tiempos_GT;  % al final tengo vector de tiempos ej: (1xNumero_Ventanas_GT) - ultima ventana
        Vector_EMG_Puntos_GT(1,window_n)=Puntos_GT;    % al final tengo vector de puntos ej: (1xNumero_Ventanas_GT) - ultima ventana
        fin_del_juego=1;                               % bandera fin de juego
        window_n=0;                                    % reinicio cont ventana
        disp('fin del juego')
    else % EN ESTE ELSE se hace el barrido de ventanas, y se concatena datos en vectores
        Vector_EMG_Tiempos_GT(1,window_n)=Tiempos_GT;  %dato escalar guardo en vector
        Vector_EMG_Puntos_GT(1,window_n)=Puntos_GT;    %dato escalar guardo en vector
        fin_del_juego=0;
    end
    
    %new_state = getState(state, action);
    new_state = table2array(Features_GT);
    
             %disp("etiquetas");disp(etiquetas);
         %disp(size(etiquetas))
         assignin('base','etiquetas',etiquetas);
         %disp("numIteration");disp(numIteration);
         assignin('base','numIteration',numIteration);
         etiqueta_actual=etiquetas(numIteration,1);                           % ground truth de cada ventana EMG - CAMBIAR - 
         %disp("etiqueta_actual");disp(etiqueta_actual);
         assignin('base','etiqueta_actual',etiqueta_actual);
         
        acciones_predichas_actual=acciones_predichas_vector(numIteration,1); % accion predicha por ANN
    
        
    %reward = getReward(new_state);
    %reward = getReward_emg_0reward(acciones_predichas_actual, etiqueta_actual); % AQUI HAY QUE DEFINIR RECOMPENSAS EN BASE A GROUNDTRUTH EMG
    if Reward_type ==true
        %disp('-1 reward')
        reward = getReward_emg(acciones_predichas_actual, etiqueta_actual); % AQUI HAY QUE DEFINIR RECOMPENSAS EN BASE A GROUNDTRUTH EMG
    else
        %disp('0 reward')
        reward = getReward_emg_0reward(acciones_predichas_actual, etiqueta_actual);
    end
    
    
    
    %     if strcmp(typeControl, 'AI')
    %         displayWorld(new_state);
    %     else
    %         displayWorld(new_state, false);
    %     end
    
    %title(['Trial = ' num2str(stepNum) ' of ' num2str(maxNumSteps)]);
    %pause(0.10);
    %     if reward == +10
    %         result = 1;
    %         title('Game over: Your agent WON :)');
    %         drawnow;
    %         break;
    %     elseif reward == -10
    %         result = 0;
    %         title('Game over: Your agent LOST :(');
    %         drawnow;
    %         break;
    %     end
    %     if stepNum >= maxNumSteps
    %         result = 0;
    %         title('Game over: Your agent LOST :(');
    %         drawnow;
    %         break;
    %     end
    
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
            TimePoints_vector(1,n1)=stride*numIteration+EMG_window_size/2;           %necesito dato de stride y tamaño de ventana de Victor
            
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
       class_result=mode(categorical(etiquetas_labels_predichas_vector_without_NoGesture));  %Saco la moda de las etiquetas dif a no gesture
%-----------------------------------------------------------------
    
            disp("Numero_Ventanas_GT");disp(Numero_Ventanas_GT-1)
        assignin('base','Numero_Ventanas_GT',Numero_Ventanas_GT-1);
        if  numIteration == Numero_Ventanas_GT -1 %==maxWindowsAllowed % reward == -10  end the game - lose
            
            %-----------check de las variables predichas que entran a eval de reconocimiento
            %disp('etiquetas_labels_predichas_vector');
            %disp(etiquetas_labels_predichas_vector); %[N,1] %vector Full
            assignin('base','etiquetas_labels_predichas_vector',etiquetas_labels_predichas_vector);
            var1=size(etiquetas_labels_predichas_vector);
            %etiquetas_labels_predichas_matrix(:,conta)=etiquetas_labels_predichas_vector;
            
            %Este lazo completa el vector etiquetas_labels_predichas_vector de ser neceario
            %, ya que tiene q coincidir con la
            %dimension del vector etiquetas_labels_predichas_matrix para poder imprimirlo en cvs
            if var1(1,1) == 39
                etiquetas_labels_predichas_vector(40,1)=("N/A");
                etiquetas_labels_predichas_vector(41,1)=("N/A");
                etiquetas_labels_predichas_vector(42,1)=("N/A");
                etiquetas_labels_predichas_matrix(:,conta)=etiquetas_labels_predichas_vector;
            elseif var1(1,1) == 40
                etiquetas_labels_predichas_vector(41,1)=("N/A");
                etiquetas_labels_predichas_vector(42,1)=("N/A");
                etiquetas_labels_predichas_matrix(:,conta)=etiquetas_labels_predichas_vector;
            elseif var1(1,1) == 41
                etiquetas_labels_predichas_vector(42,1)=("N/A");
                etiquetas_labels_predichas_matrix(:,conta)=etiquetas_labels_predichas_vector;
            end
            assignin('base','etiquetas_labels_predichas_matrix',etiquetas_labels_predichas_matrix);
           
            
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
            
            
            disp('Eval Recognition');
            % GROUND TRUTH (no depende del modelo)------------
            repInfo.gestureName =  gestureName_GT; % OK -----  categorical({'waveIn'});   %CAMBIAR - poner etiqueta de muestra de señal
            assignin('base','gestureName_GT',gestureName_GT);
            repInfo.groundTruth = groundTruth_GT; %   REV -----
            assignin('base','groundTruth_GT',groundTruth_GT);
            %repInfo.groundTruth = false(1, 1000);   %Each_complete_signal;           %false(1, 1000);            %CAMBIAR
            %repInfo.groundTruth(800:1600) = true;   %CAMBIAR (64datos*40ventanas)
            
            %plot(repInfo.groundTruth)
            
            % PREDICCION--------------------------------------
            response.vectorOfLabels = categorical(etiquetas_labels_predichas_vector_simplif); % OK ----- [1,N] % categorical(etiquetas_labels_predichas_vector_simplif); % %CAMBIAR
            response.vectorOfTimePoints = TimePoints_vector; % OK -----  [40 200 400 600 800 999]; %1xw double  TimePoints_vector                %CAMBIAR
            % tiempo de procesamiento
            response.vectorOfProcessingTimes = ProcessingTimes_vector; % OK -----[0.1 0.1 0.1 0.1 0.1 0.1]; % ProcessingTimes_vector'; % [0.1 0.1 0.1 0.1 0.1 0.1]; % 1xw double                                    %CAMBIAR
            response.class =  categorical(class_result); % OK ----- categorical({'waveIn'});                %aqui tengo que usar la moda probablemente           %CAMBIAR
            
            %-----------------------------------------------
            %r1 = 1;
            r1 = evalRecognition(repInfo, response);
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
                    disp(resultGame)
                    gameOn = false;
                    reward = -10;
                    % Suma_aciertos >= 30 &&
                elseif  r1.recogResult==1 && fin_del_juego==1 % numIteration == maxIterationsAllowed % eward == +10   end the game - win
                    resultGame = 'won ';
                    disp(resultGame)
                    countWins = countWins + 1;
                    gameOn = false;
                    reward = +10;
                    
                end
                
            end
            
        end
    
        % Cumulative reward so far for the current episode
        cumulativeGameReward = cumulativeGameReward + reward;
        
        % Cumulative reward so far for the current episode
        cumulativeIterationReward = cumulativeIterationReward + reward;
        
         % Updating the state
        state = new_state;    
        
        
    % state = new_state;
    
    
    
    
end
end
