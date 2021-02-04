function [] = Code_0(emgRepetition)
addpath(genpath('Data'));
addpath(genpath('PreProcessing'));
addpath(genpath('FeatureExtraction'));
addpath(genpath('Classification'));
addpath(genpath('PostProcessing'));

rng('default');
warning off all;

syncro        = 4;  % ???
energy_umbral = 0.2; % valor para umbrales de energia de cada ventana
model=2;


modelSelected='specific';
assignin('base','modelSelected',modelSelected);


% ============================== SPECIFIC ===================================
 % dir('Data\Specific') list the objects like: .      ..     user1. With
 % attributes name, folder, date, etc in structs, and size >2
 
dataPacket      = orderfields(dir('Data\Specific'));
dataPacketSize  = length(dataPacket);
pathUser        = pwd;  % the site where the app is running
pathOrigin      = 'Data\Specific';

assignin('base','dataPacket',     dataPacket);
assignin('base','dataPacketSize', dataPacketSize);
assignin('base','pathUser',       pathUser);
assignin('base','pathOrigin',     pathOrigin);

userCounter     = 1;
% ================================ MAD & Energy ==================================
orientation  = cell(dataPacketSize-2,2);
for k=1:dataPacketSize  % each data found in Data/Specific
    
    if ~(strcmpi(dataPacket(k).name, '.') || strcmpi(dataPacket(k).name, '..'))  % the two objects by defect are excluded
        usuario     = dataPacket(k).name;
        userFolder  = horzcat(pathUser,'\',pathOrigin,'\',dataPacket(k).name,'\','userData.mat');
        % load the userData.mat
        load(userFolder);
        
        if syncro>0
            
            % found the first location of "wave out"=syncro gesture?, now
            % in location 126
            for x=1:150
                gesto_=userData.training{x,1}.gestureName;
                if gesto_=="waveOut"
                    location_=x;
                    break;
                end
            end
            elec_=zeros(1,syncro);
            aux=1;
            energy_order=zeros(syncro,8);
            % =======================================================
            %                     WITH ROTATION
            % =======================================================
            for goto_=location_:location_+syncro-1  % range to fill elec_
                
                % =============================================================
                if goto_==location_
                    % simulateRotation() is a function that generates
                    % [-3,-2,-1,0,1,2,3] -> [5,6,7,0,1,2,3] orden randomly
                    assignin('base','simulate_Rotation', simulateRotation());
                end
                simulate_Rotation       = evalin('base','simulate_Rotation');
                % ============================================================
                % reads the signal and interchange chanels using an array with the rotation (:, [6 7 0 1 ...])
                emgData             = userData.training{goto_,1}.emg(:,simulate_Rotation);  
                % groundTruthIndex: range where groundtruth is
                Index_              = userData.training{goto_,1}.groundTruthIndex;  
                Index_high_         = Index_(1,2);
                % 255? that means that the mean length of groundtruth is
                % 255? should be the distance between index(1) and index(2)
                % emgData only takes the 8 channels of the "groundtruth"
                emgData             = emgData(Index_high_ - 255:Index_high_,:);
                energy_wm           = WMoos_F5(emgData');
                energy_order(aux,:) = energy_wm;
                [~,max_energy]      = max((energy_wm));
                elec_(1,aux)        = max_energy;
                aux = aux+1;
            end
            ref_partial         = histcounts(elec_(1,:),1:(8+1));
            [~,ref]             = max(ref_partial);
            xyz                 = ref;
        else
            xyz_aux             = simulateRotation();
            xyz                 = xyz_aux(:,1);
        end
        % ================== Umbral =========================
        
        calibration_umbral=zeros(8,syncro);
        for o=1:syncro
            waveout_pure=userData.sync{o,1}.emg(:,simulate_Rotation);
            umbral_envelope_wm=WMoos_F5(waveout_pure');
            calibration_umbral(:,o)=umbral_envelope_wm;
        end
        sequence_=WM_X(xyz);
        calibration_umbral=calibration_umbral';
        calibration_umbral=calibration_umbral(:,sequence_);
        mean_umbral=calibration_umbral;
        mean_umbral=mean(mean_umbral,1);
        val_umbral_high = energy_umbral*sum(mean_umbral(1:4))/4;
        val_umbral_low  = energy_umbral*sum(mean_umbral(5:8))/4;
        
        % ==================================================
        
        orientation{userCounter,1} = usuario;
        orientation{userCounter,2} = xyz;
        orientation{userCounter,3} = val_umbral_low;
        orientation{userCounter,4} = val_umbral_high;
        orientation{userCounter,5} = simulate_Rotation(:,1);
        
        RepTraining = evalin('base','RepTraining');
        noGestureDetection = evalin('base','noGestureDetection');
        randomGestures = evalin('base','randomGestures');
        on  = true;
        rangeValues = evalin('base','rangeValues');
        packetEMG   = evalin('base','packetEMG');

        
        if randomGestures == on
            % Random values
            
            if noGestureDetection==on
                %noGesture detection
                
                if rangeValues <=150
                    emg_rand=1:1:rangeValues;
                    emg_rand=emg_rand(randperm(length(emg_rand)));
                    
                    if packetEMG == true
                        
                        y1 =  1:1:25;
                        y1 = y1(randperm(length(y1)));
                        y2 = 26:1:50;
                        y2 = y2(randperm(length(y2)));
                        y3 = 51:1:75;
                        y3 = y3(randperm(length(y3)));
                        y4 = 76:1:100;
                        y4 = y4(randperm(length(y4)));
                        y5 =101:1:125;
                        y5 = y5(randperm(length(y5)));
                        y6 =126:1:150;
                        y6 = y6(randperm(length(y6)));
                        
                        emg_rand =[y1,y2,y3,y4,y5,y6];
                        
                        
                        ay=0;
                        for x=1:1:25
                            
                            x1=emg_rand(x);
                            x2=emg_rand(x+25);
                            x3=emg_rand(x+50);
                            x4=emg_rand(x+75);
                            x5=emg_rand(x+100);
                            x6=emg_rand(x+125);
                            
                            ax=[x1,x2,x3,x4,x5,x6];
                            ay=horzcat(ay,ax);
                            
                        end
                        ay=ay(2:end);
                        emg_rand=ay(1:rangeValues);
                    end
                    
                    
                else
                    emg_rand=1:1:150;
                    emg_rand=emg_rand(randperm(length(emg_rand)));
                    emg_rand_aux=151:1:rangeValues;
                    emg_rand_aux=emg_rand_aux(randperm(length(emg_rand_aux)));
                    emg_rand=horzcat(emg_rand,emg_rand_aux);
                    
   
                    if packetEMG == true
                        
                        y1 =  1:1:25;
                        y1 = y1(randperm(length(y1)));
                        y2 = 26:1:50;
                        y2 = y2(randperm(length(y2)));
                        y3 = 51:1:75;
                        y3 = y3(randperm(length(y3)));
                        y4 = 76:1:100;
                        y4 = y4(randperm(length(y4)));
                        y5 =101:1:125;
                        y5 = y5(randperm(length(y5)));
                        y6 =126:1:150;
                        y6 = y6(randperm(length(y6)));
                        
                        emg_rand =[y1,y2,y3,y4,y5,y6];

                        ay=0;
                        for x=1:1:25
                            
                            x1=emg_rand(x);
                            x2=emg_rand(x+25);
                            x3=emg_rand(x+50);
                            x4=emg_rand(x+75);
                            x5=emg_rand(x+100);
                            x6=emg_rand(x+125);
                            
                            ax=[x1,x2,x3,x4,x5,x6];
                            ay=horzcat(ay,ax);
                            
                        end
                        ay_=ay(2:end);                        
                        
                        
                        y1 = 151:1:175;
                        y1 = y1(randperm(length(y1)));
                        y2 = 176:1:200;
                        y2 = y2(randperm(length(y2)));
                        y3 = 201:1:225;
                        y3 = y3(randperm(length(y3)));
                        y4 = 226:1:250;
                        y4 = y4(randperm(length(y4)));
                        y5 =251:1:275;
                        y5 = y5(randperm(length(y5)));
                        y6 =276:1:300;
                        y6 = y6(randperm(length(y6)));
                        
                        emg_rand =[y1,y2,y3,y4,y5,y6];                     
                        emg_rand=horzcat(zeros(1,150),emg_rand);

                        
                        ay=0;
                        for x=1:1:25
                            
                            x1=emg_rand(x+150);
                            x2=emg_rand(x+175);
                            x3=emg_rand(x+200);
                            x4=emg_rand(x+225);
                            x5=emg_rand(x+250);
                            x6=emg_rand(x+275);
                            
                            ax=[x1,x2,x3,x4,x5,x6];
                            ay=horzcat(ay,ax);
                            
                        end
                        ay=ay(2:end);
                        ay=horzcat(ay_,ay);
                        emg_rand=ay(1:rangeValues);
                        
                    end
          
                    
                end
                
                
            else
               
                %noGesture no detection
                
                if rangeValues<=150
                    emg_rand=26:1:rangeValues;
                    emg_rand=emg_rand(randperm(length(emg_rand)));
                    emg_rand_aux=zeros(1,25);
                    emg_rand=horzcat(emg_rand_aux,emg_rand);                    
                    
                    
                                        
                    if packetEMG == true                        

                        y2 = 26:1:50;
                        y2 = y2(randperm(length(y2)));
                        y3 = 51:1:75;
                        y3 = y3(randperm(length(y3)));
                        y4 = 76:1:100;
                        y4 = y4(randperm(length(y4)));
                        y5 =101:1:125;
                        y5 = y5(randperm(length(y5)));
                        y6 =126:1:150;
                        y6 = y6(randperm(length(y6)));
                        
                        emg_rand =[y2,y3,y4,y5,y6];                        
                        
                        ay=0;
                        for x=1:1:25                         
                            
                            x2=emg_rand(x);
                            x3=emg_rand(x+25);
                            x4=emg_rand(x+50);
                            x5=emg_rand(x+75);
                            x6=emg_rand(x+100);                            
                            ax=[x2,x3,x4,x5,x6];
                            ay=horzcat(ay,ax);                            
                        end
                        ay=ay(2:end);
                        emg_rand=horzcat(zeros(1,25),ay);
                        emg_rand=emg_rand(1:rangeValues);
                    end    
                    
                else
                    emg_rand=26:1:150;
                    emg_rand=emg_rand(randperm(length(emg_rand)));
                    emg_rand_aux=zeros(1,25);
                    emg_rand=horzcat(emg_rand_aux,emg_rand);
                    
                    
                    emg_rand_=176:1:rangeValues;
                    emg_rand_=emg_rand_(randperm(length(emg_rand_)));
                    emg_rand_aux=zeros(1,25);
                    emg_rand_=horzcat(emg_rand_aux,emg_rand_);
                    
                    emg_rand=horzcat(emg_rand,emg_rand_);
                    
                    
                    
                    
                    
                    if packetEMG == true
                        
                        y2 = 26:1:50;
                        y2 = y2(randperm(length(y2)));
                        y3 = 51:1:75;
                        y3 = y3(randperm(length(y3)));
                        y4 = 76:1:100;
                        y4 = y4(randperm(length(y4)));
                        y5 =101:1:125;
                        y5 = y5(randperm(length(y5)));
                        y6 =126:1:150;
                        y6 = y6(randperm(length(y6)));
                        
                        emg_rand =[y2,y3,y4,y5,y6];
                        
                        ay=0;
                        for x=1:1:25
                            
                            x2=emg_rand(x);
                            x3=emg_rand(x+25);
                            x4=emg_rand(x+50);
                            x5=emg_rand(x+75);
                            x6=emg_rand(x+100);
                            ax=[x2,x3,x4,x5,x6];
                            ay=horzcat(ay,ax);
                        end
                        ay=ay(2:end);
                        y_=horzcat(zeros(1,25),ay);
                        
                        
                        
                        
                        y2 = 176:1:200;
                        y2 = y2(randperm(length(y2)));
                        y3 = 201:1:225;
                        y3 = y3(randperm(length(y3)));
                        y4 = 226:1:250;
                        y4 = y4(randperm(length(y4)));
                        y5 = 251:1:275;
                        y5 = y5(randperm(length(y5)));
                        y6 = 276:1:300;
                        y6 = y6(randperm(length(y6)));
                        
                        emg_rand =[y2,y3,y4,y5,y6];
                        
                        ay=0;
                        for x=1:1:25
                            
                            x2=emg_rand(x);
                            x3=emg_rand(x+25);
                            x4=emg_rand(x+50);
                            x5=emg_rand(x+75);
                            x6=emg_rand(x+100);
                            ax=[x2,x3,x4,x5,x6];
                            ay=horzcat(ay,ax);
                        end
                        ay=ay(2:end);
                        yy_=horzcat(zeros(1,25),ay);
                        
                        emg_rand=horzcat(y_,yy_);
                        emg_rand=emg_rand(1:rangeValues);                
                        
                    end     
                    
                end
                
                
            end
            
        else
            % No Random values
            
            if noGestureDetection==on
                %noGesture detection
                if rangeValues<=150
                    emg_rand=1:1:rangeValues;
                    
                    if packetEMG == true
                        ay=0;
                        for x=1:1:25
                            
                            x1=emg_rand(x);
                            x2=emg_rand(x+25);
                            x3=emg_rand(x+50);
                            x4=emg_rand(x+75);
                            x5=emg_rand(x+100);
                            x6=emg_rand(x+125);
                            
                            ax=[x1,x2,x3,x4,x5,x6];
                            ay=horzcat(ay,ax);
                            
                        end
                        ay=ay(2:end);
                        emg_rand=ay(1:rangeValues);
                    end
                    
                else
                    emg_rand=1:1:rangeValues;
                    if packetEMG == true
                        ay=0;
                        for x=1:1:25
                            
                            x1=emg_rand(x);
                            x2=emg_rand(x+25);
                            x3=emg_rand(x+50);
                            x4=emg_rand(x+75);
                            x5=emg_rand(x+100);
                            x6=emg_rand(x+125);
                            
                            ax=[x1,x2,x3,x4,x5,x6];
                            ay=horzcat(ay,ax);
                            
                        end
                        ay_=ay(2:end);
                       
                        ay=0;
                        for x=1:1:25
                            
                            x1=emg_rand(x+150);
                            x2=emg_rand(x+175);
                            x3=emg_rand(x+200);
                            x4=emg_rand(x+225);
                            x5=emg_rand(x+250);
                            x6=emg_rand(x+275);
                            
                            ax=[x1,x2,x3,x4,x5,x6];
                            ay=horzcat(ay,ax);
                            
                        end
                        ay=ay(2:end);
                        ay=horzcat(ay_,ay);
                        emg_rand=ay(1:rangeValues);
                        
                    end
                    
                    
                end 
      
            else
                %noGesture no detection
                if rangeValues <=150
                    emg_rand=26:1:rangeValues;
                    emg_rand_aux=zeros(1,25);
                    emg_rand=horzcat(emg_rand_aux,emg_rand);
                    
                    
                    
                    if packetEMG == true
                        ay=0;
                        for x=1:1:25
                            
                            x2=emg_rand(x+25);
                            x3=emg_rand(x+50);
                            x4=emg_rand(x+75);
                            x5=emg_rand(x+100);
                            x6=emg_rand(x+125);
                            
                            ax=[x2,x3,x4,x5,x6];
                            ay=horzcat(ay,ax);
                            
                        end
                        ay=ay(2:end);
                        emg_rand=ay(1:rangeValues-25);
                        emg_rand=horzcat(zeros(1,25),emg_rand);
                    end
                        
                    
                else
              
                     
                    emg_rand=26:1:150;
                    emg_rand_aux=zeros(1,25);
                    emg_rand=horzcat(emg_rand_aux,emg_rand);                    
                    
                    emg_rand_=176:1:rangeValues;
                    emg_rand_aux=zeros(1,25);
                    emg_rand_=horzcat(emg_rand_aux,emg_rand_);
                    
                    emg_rand=horzcat(emg_rand,emg_rand_); 
                    
                    
                    if packetEMG == true
                        ay=0;
                        for x=1:1:25
                            
                            
                            x2=emg_rand(x+25);
                            x3=emg_rand(x+50);
                            x4=emg_rand(x+75);
                            x5=emg_rand(x+100);
                            x6=emg_rand(x+125);
                            
                            ax=[x2,x3,x4,x5,x6];
                            ay=horzcat(ay,ax);
                            
                        end
                        ay=ay(2:end);
                        ay_=horzcat(zeros(1,25),ay);
                        
                        ay=0;
                        for x=1:1:25                           
                            
                            x2=emg_rand(x+175);
                            x3=emg_rand(x+200);
                            x4=emg_rand(x+225);
                            x5=emg_rand(x+250);
                            x6=emg_rand(x+275);
                            
                            ax=[x2,x3,x4,x5,x6];
                            ay=horzcat(ay,ax);
                            
                        end
                        ay=ay(2:end);
                        ay=horzcat(zeros(1,25),ay);
                        ay=horzcat(ay_,ay);
                        emg_rand=ay(1:rangeValues);
                        
                    end
                       
                end   
            end
            
        end
        
        orientation{userCounter,6} = emg_rand;
        userCounter = userCounter+1;  %only users differents from . and ..
        
    end
end
% in orientation there are orientation and  the index of the data to beasfd read
% each row is the data of a user
assignin('base','orientation',orientation);

% ============================ Initialization =============================

userIndex       = 3;            % Start reading from first user, the first user is in position 3
%emgRepetition   = 1;            % Start getting the i-th gesture repetition
stepControl     = 1;            % Set the first windows index(steps) to 1
responseIndex   = 1;            % First response goes to the location 1
responseBuffer  = 'noGesture';  % First gesture located the buffer is "noGesture"

assignin('base','JP_Longitud',1);

assignin('base','userIndex',     userIndex);
assignin('base','emgRepetition', emgRepetition);
assignin('base','stepControl',   stepControl);
assignin('base','responseIndex', responseIndex);
assignin('base','responseBuffer',responseBuffer);

assignin('base','low_umbral', 0);
assignin('base','high_umbral',0);
assignin('base','matrix',0);

assignin('base','sequence_mad', true);
assignin('base','change_user',    false);

% this index is of the orientation, and is index "true"
assignin('base','index_user', userIndex-2);  % Just begin in 3?
testControl = true;
assignin('base','testControl', testControl);  % ?????????

counter=1;
assignin('base','counter', counter);
assignin('base','EMG_Activity',"noGesture");

end

