function gt_gestures_labels = mapGroundTruthToLabels(numWindows, emg_window_size, stride, groundTruthIndex_GT, gestureName_GT)
%MAPGROUNDTRUTHTOLABELS 
%Creo vector de etiquetas para Ground truth x ventana
% if the window is holding a EMG_window_size/5 part at least of the groundtruth, is labeled as the gesture.

%Creo vector con limite derecho de cada ventana. Ejm: win=200,stride=30 -> 200 230 260.... 980
gt_gestures_pts=zeros(1, numWindows); %% CORREGIR
gt_gestures_pts(1,1)=emg_window_size;
for k = 1:numWindows-1
    gt_gestures_pts(1,k+1)=gt_gestures_pts(1,k)+stride;
end
% assignin('base','gt_gestures_pts',gt_gestures_pts); DELETE?
            
gt_gestures_labels=strings;

for k2 = 1:numWindows
    % PENDING: por qué para 5? EMG_window_size/5
    if gt_gestures_pts(1,k2) > (groundTruthIndex_GT(1,1) + emg_window_size/5) && gt_gestures_pts(1,k2) < groundTruthIndex_GT(1,2)
        gt_gestures_labels(1,k2)=string(gestureName_GT);
    elseif  gt_gestures_pts(1,k2)-emg_window_size < (groundTruthIndex_GT(1,2)-emg_window_size/5 ) && gt_gestures_pts(1,k2) > groundTruthIndex_GT(1,2)
        gt_gestures_labels(1,k2)=string(gestureName_GT);
    else
        gt_gestures_labels(1,k2)="noGesture";
    end
end

end

