function showFigureAccuracyByEpisode(dir_results, filename_experiment, ...
        list_users, ...
        accuracy_by_episode_training, ...
        accuracy_by_episode_validation, ...
        accuracy_by_episode_testing, ...
        RepTraining, repTrainingForValidation, repTrainingForTesting, ...
        make_validation, make_testing, id_figure ...
)
%SHOWFIGURES Summary of this function goes here
%   Detailed explanation goes here

num_users = length(list_users);


% Accuracy window de la última época de training y testing
accuracy_training_usersXepisodes = reshape(accuracy_by_episode_training(end,:), num_users, RepTraining);
mean_accuracy_training_episodes = mean(accuracy_training_usersXepisodes, 1);

figure(id_figure);
hold on;
plot(1:length(mean_accuracy_training_episodes), mean_accuracy_training_episodes, 'b');

if make_validation
   
    accuracy_validation_usersXepisodes = reshape(accuracy_by_episode_validation(end,:), num_users, repTrainingForValidation);

    mean_accuracy_validation_episodes = mean(accuracy_validation_usersXepisodes, 1);
    plot(1:length(mean_accuracy_validation_episodes), mean_accuracy_validation_episodes, 'r');
end

if make_testing
   
    accuracy_testing_usersXepisodes = reshape(accuracy_by_episode_testing(end,:), num_users, repTrainingForTesting);

    mean_accuracy_testing_episodes = mean(accuracy_testing_usersXepisodes, 1);
    plot(1:length(mean_accuracy_testing_episodes), mean_accuracy_testing_episodes, 'g');
end

hold off;
xlabel('Episode (final epoch)')
ylabel('Accuracy window')
legend('Training', 'Validation', 'Testing')
title('Accuracy window in last epoch')

saveas(gcf, dir_results + "/Figure_"+filename_experiment+"_accMeanByEpisodes.png");
        


end

