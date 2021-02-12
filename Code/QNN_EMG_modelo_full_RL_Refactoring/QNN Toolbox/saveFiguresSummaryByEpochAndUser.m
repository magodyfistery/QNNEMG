function saveFiguresSummaryByEpochAndUser(dir_results, filename_experiment, ...
        list_users, ...
        recognition_accuracy_training, ...
        classif_mode_ok_accuracy_training, ...
        classif_by_window_accuracy_training, ...
        recognition_accuracy_validation, ...
        classif_mode_ok_accuracy_validation, ...
        classif_by_window_accuracy_validation, ...
        recognition_accuracy_testing, ...
        classif_mode_ok_accuracy_testing, ...
        classif_by_window_accuracy_testing, ...
        make_validation, make_testing ...
)
%SHOWFIGURES Summary of this function goes here
%   Detailed explanation goes here

num_users = length(list_users);
    
for index_id_user=1:num_users

    figure(1);
    subplot(3,1,1);
    hold on;
    plot(1:length(recognition_accuracy_training(:, index_id_user)),recognition_accuracy_training(:, index_id_user), 'b');
    if make_validation
        plot(1:length(recognition_accuracy_validation(:, index_id_user)),recognition_accuracy_validation(:, index_id_user), 'r');
    end
    if make_testing
        plot(1:length(recognition_accuracy_testing(:, index_id_user)),recognition_accuracy_testing(:, index_id_user), 'g');
    end
    hold off;
    xlabel('Epoch')
    ylabel('Recognition Accuracy')
    legend('Training', 'Validation', 'Testing')
    title('Recognition Accuracy for epoch')

    subplot(3,1,2);
    hold on;
    plot(1:length(classif_mode_ok_accuracy_training(:, index_id_user)),classif_mode_ok_accuracy_training(:, index_id_user), 'b');
    if make_validation
        plot(1:length(classif_mode_ok_accuracy_validation(:, index_id_user)),classif_mode_ok_accuracy_validation(:, index_id_user), 'r');
    end
    if make_testing
        plot(1:length(classif_mode_ok_accuracy_testing(:, index_id_user)),classif_mode_ok_accuracy_testing(:, index_id_user), 'g');
    end
    
    hold off;
    xlabel('Epoch')
    ylabel('Classification Accuracy')
    legend('Training', 'Validation', 'Testing')
    title('Classification Accuracy for epoch')

    subplot(3,1,3);
    hold on;
    plot(1:length(classif_by_window_accuracy_training(:, index_id_user)),classif_by_window_accuracy_training(:, index_id_user), 'b');
    
    if make_validation
        plot(1:length(classif_by_window_accuracy_validation(:, index_id_user)),classif_by_window_accuracy_validation(:, index_id_user), 'r');
    end
    if make_testing
        plot(1:length(classif_by_window_accuracy_testing(:, index_id_user)),classif_by_window_accuracy_testing(:, index_id_user), 'g');
    end
    
    hold off;
    xlabel('Epoch')
    ylabel('Window Accuracy')
    legend('Training', 'Validation', 'Testing')
    title('Window Accuracy for epoch')


    saveas(gcf, dir_results + "/Figures_summary_byUserEpochs/Figure_summary_"+filename_experiment+"_user"+list_users(index_id_user)+".png");
    close figure 1;

end

end

