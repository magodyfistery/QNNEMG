function [Numero_Ventanas,EMG,Vector_EMG_Features,Vector_EMG_Tiempos,Vector_EMG_Puntos,Nombre_Usuario,gestureName_,groundTruthIndex_,groundTruth_x] = Code_1(orientation,dataPacketSize,RepTraining,RepTotalTraining, verbose_level)


 Numero_Ventanas      = 0;
 EMG                  = 0;
 Vector_EMG_Features  = 0;
 Vector_EMG_Tiempos   = 0;
 Vector_EMG_Puntos    = 0 ;
 Nombre_Usuario       = 'NaN';
 gestureName_         = 'NaN';
 groundTruthIndex_    = [0 1];
 groundTruth_x        = logical(zeros(1,1000));


testControl = evalin('base', 'testControl');  % :boolean
Stride = evalin('base', 'Stride');
WindowsSize = evalin('base', 'WindowsSize');


if testControl==true
    
    % when executes Code_0:
    % change user is false, userIndex is 3, testControl is true always
    change_user = evalin('base', 'change_user');  % :boolean
    userIndex   = evalin('base', 'userIndex');
    
    if ~change_user
        index_user = evalin('base', 'index_user');
        usuario           = orientation{index_user,1};
        index_user        = index_user+1;
        assignin('base','change_user',true);
    end
    
    % emgDataTested is the window with dims: window size (points)x8
    % this code forwards the windows stride by stride each time is called
    emgDataTested     = OfflineDataExp4S(1,Stride,"on", verbose_level);
    % Always are 40 features, independently of window size or stride
    emgFeatures       = getFeatures(emgDataTested);
    % when control Exit =1, means that was the last window
    controlExit       = evalin('base','controlExit');
    % if change user is true, this update the value
    usuario           = evalin('base','usuario');
    % emgRepetition is the value of the random data like 26,51,etc
    emgRepetition     = evalin('base','emgRepetition');   
    
    
    timeAnalized            = evalin('base','timeAnalized');
    WindowRep               = evalin('base','JP_Longitud');

    % the total windows are 1 + (WindowRep - WindowsSize)/Stride
    % but there is a last part not taken in account, for this is the
    % floor function ex: 17.4 -> 17 if window size is 100, then 40
    % points are discarded. This is good or bad??????
    % Corrected formula:  1+ floor((WindowRep - WindowsSize)/Stride)
    % they forget to include the first window
    Numero_Ventanas     = 1+ floor((WindowRep - WindowsSize)/Stride); % *******************************************
    EMG                 = emgDataTested;                % EMG de Window_size puntos
    Vector_EMG_Features = emgFeatures;                  % Vectores en formato de tabla
   %Vector_EMG_Features = table2array(emgFeatures);     % Vectores en formato de matriz
    Vector_EMG_Tiempos  = timeAnalized;
    Vector_EMG_Puntos   = timeAnalized*200;  % SHOULD CHANGE 200 by window SIZE?
    Nombre_Usuario      = usuario;

    gestureName_         = evalin('base','gestureName_');

    if gestureName_~= "noGesture" && gestureName_ ~= "unknown"
        groundTruthIndex_    = evalin('base','groundTruthIndex_');
        groundTruth_x        = evalin('base','groundTruth_x');
    end
    
    if controlExit==true
        
        %clc
        assignin('base','emgRepetition',emgRepetition+1);        
        counter = evalin('base','counter');
        counter=counter+1;
        assignin('base','counter',counter);
      
        
        if counter<RepTraining+1
            %clc
            %disp('Fin de la muestra, 40 ventanas')
            %pause(2)

        else
            
            assignin('base','emgRepetition',301);
            counter=301;
            assignin('base','counter',301);
                 
        end
        
        if counter==301
            %clc
            %disp('Fin de la muestra, 40 ventanas')
            %disp('Cambio de usuario')
            assignin('base','emgRepetition', evalin('base','rangeDown'));
            assignin('base','counter',1); 
            assignin('base', 'change_user', true);
            
            %pause(2)
        end
     
    end
        
    testControl    = evalin('base','testControl');
end

end

