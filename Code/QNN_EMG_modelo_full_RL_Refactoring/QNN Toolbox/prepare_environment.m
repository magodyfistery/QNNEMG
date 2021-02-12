function prepare_environment(params, verbose_level)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

% Paths to other codes
addpath('Multivariate Regression Neural Network Toolbox');

addpath(genpath('FeatureExtraction'));
addpath(genpath('Data'));
addpath(genpath('PreProcessing'));
addpath(genpath('testingJSON'));
addpath(genpath('trainingJSON'));
addpath(genpath('utils'));


on  = true;
off = false;


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
    

assignin('base','WindowsSize',  params.window_size);
assignin('base','Stride',  params.stride);

%==============Parameters for Code_0 (preprocesser of emg data)==========

assignin('base','post_processing',     on);   %on si quiero post procesamiento en vector de etiquetas resultadnte                                          %off si quiero solo recomp -10 x recog 
 
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

end

