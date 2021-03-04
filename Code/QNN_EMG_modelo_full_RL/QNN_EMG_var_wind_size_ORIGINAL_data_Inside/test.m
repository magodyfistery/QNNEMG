
EMG_window_size = 200;
Stride = 100;
gestureName_GT = "GESTO";
Numero_Ventanas_GT = 8;
gt_gestures_pts=zeros(1,Numero_Ventanas_GT);
gt_gestures_pts(1,1)=EMG_window_size;
groundTruthIndex_GT = [500 700];

for k = 1:Numero_Ventanas_GT-1
    gt_gestures_pts(1,k+1)=gt_gestures_pts(1,k)+Stride;
end

gt_gestures_labels=strings;

for k2 = 1:Numero_Ventanas_GT
    if gt_gestures_pts(1,k2) > (groundTruthIndex_GT(1,1) + EMG_window_size/5) && gt_gestures_pts(1,k2) < groundTruthIndex_GT(1,2)
        %disp('case1')
        gt_gestures_labels(1,k2)=string(gestureName_GT);
    elseif  gt_gestures_pts(1,k2)-EMG_window_size < (groundTruthIndex_GT(1,2)-EMG_window_size/5 ) && gt_gestures_pts(1,k2) > groundTruthIndex_GT(1,2)
        %disp('case2')
        gt_gestures_labels(1,k2)=string(gestureName_GT);
    else
        %disp('case3')
        gt_gestures_labels(1,k2)="noGesture";
    end
end
    