function [training_accuracy, validation_accuracy, testing_accuracy, qnn] ...
    = QNN_emg_Exp_Replay(params, verbose_level, RepTraining, ...
    repTrainingForValidation, repTrainingForTesting, ...
    dir_specific_experiment, filename_experiment, excel_dir, list_users, ...
    make_validation, make_testing, user_reset)
%{

%}
    
validation_accuracy = -1;
testing_accuracy = -1;

rangeDown = 26;
prepare_environment(params, verbose_level-1);
assignin('base','RepTraining',  RepTraining); % initial value
Code_0(rangeDown);
orientation      = evalin('base', 'orientation');
dataPacket = evalin('base','dataPacket');
num_users = length(list_users);

rangeDown_validation=rangeDown+RepTraining;
rangeDown_testing=rangeDown_validation+repTrainingForValidation;

% variables for plotting
accuracy_by_episode_training = zeros(params.numEpochs, RepTraining, num_users);

recognition_accuracy_training = zeros(params.numEpochs, num_users);
classif_mode_ok_accuracy_training = zeros(params.numEpochs, num_users);
classif_by_window_accuracy_training = zeros(params.numEpochs, num_users);

accuracy_by_episode_validation = zeros(params.numEpochs, repTrainingForValidation, num_users);
recognition_accuracy_validation = zeros(params.numEpochs, num_users);
classif_mode_ok_accuracy_validation = zeros(params.numEpochs, num_users);
classif_by_window_accuracy_validation = zeros(params.numEpochs, num_users);
    
accuracy_by_episode_testing = zeros(params.numEpochs, repTrainingForTesting, num_users);
recognition_accuracy_testing = zeros(params.numEpochs, num_users);
classif_mode_ok_accuracy_testing = zeros(params.numEpochs, num_users);
classif_by_window_accuracy_testing = zeros(params.numEpochs, num_users);

% QNN Configuration
qnnOption = QNNOption(params.numNeuronsLayers, params.transferFunctions, ...
                params.lambda, params.learningRate, params.numEpochsToIncreaseMomentum, ...
                params.momentum, params.initialMomentum, ...
                params.miniBatchSize, params.gamma, params.epsilon);

qnn = QNN(qnnOption, params.rewardType, params.reserved_space_for_gesture);
qnn.initTheta(initWeightsOptimiced(qnn.qnnOption.numNeuronsLayers));



t_start = tic;
for epoch=1:params.numEpochs
    if verbose_level >= 1
        fprintf("****************\nEpoch: %d of %d\n***************\n", epoch, params.numEpochs);
    end
    
    if user_reset
        

        for index_id_user=1:num_users
            % TRAINING
            qnn = QNN(qnnOption, params.rewardType, params.reserved_space_for_gesture);
            qnn.initTheta(initWeightsOptimiced(qnn.qnnOption.numNeuronsLayers));
            
            assignin('base','RepTraining',  RepTraining);
            
            user_folder = "user"+list_users(index_id_user);

            userData = loadSpecificUserByName(user_folder);
            index_in_packet = getUserIndexInPacket(dataPacket, user_folder);


            
            assignin('base', 'userIndex', index_in_packet);
            assignin('base','rangeDown', rangeDown);
            assignin('base','emgRepetition', rangeDown);

            [summary_episodes, summary_classifications_mode, wins_by_episode, ...
                loses_by_episode] = qnn.train(params.window_size, ...
                                    params.stride, userData, orientation, ...
                                    verbose_level-1, RepTraining);


            recognition_accuracy_training(epoch, index_id_user) = summary_episodes(1)/(summary_episodes(1) + summary_episodes(2));
            classif_mode_ok_accuracy_training(epoch, index_id_user) = summary_classifications_mode(1)/(summary_classifications_mode(1)+summary_classifications_mode(2));
            classif_by_window_accuracy_training(epoch, index_id_user) = sum(wins_by_episode)/(sum(wins_by_episode)+sum(loses_by_episode));

            accuracy_by_episode_training(epoch, :, index_id_user) = wins_by_episode ./ (wins_by_episode + loses_by_episode);

            if make_validation
                % VALIDATION
                % The validation is made with theta trained with all users
                % and for each epoch. It should increase
                assignin('base','RepTraining',  repTrainingForValidation); % PENDING: revisar si esto hace algo


                assignin('base','rangeDown', rangeDown_validation); 
                assignin('base','emgRepetition', rangeDown_validation);



                [validation_summary_episodes, validation_summary_classifications_mode,...
                    validation_wins_by_episode, validation_loses_by_episode] = ...
                    qnn.test(params.window_size, params.stride, userData, ...
                    orientation, verbose_level-1, repTrainingForValidation, true);


                recognition_accuracy_validation(epoch, index_id_user) = validation_summary_episodes(1)/(validation_summary_episodes(1) + validation_summary_episodes(2));
                classif_mode_ok_accuracy_validation(epoch, index_id_user) = validation_summary_classifications_mode(1)/(validation_summary_classifications_mode(1)+validation_summary_classifications_mode(2));
                classif_by_window_accuracy_validation(epoch, index_id_user) = sum(validation_wins_by_episode)/(sum(validation_wins_by_episode)+sum(validation_loses_by_episode));

                accuracy_by_episode_validation(epoch, :, index_id_user) = validation_wins_by_episode ./ (validation_wins_by_episode + validation_loses_by_episode);
            end
            
            if make_testing

                % TESTING
                % The test is made with theta trained with all users
                % and for each epoch. It should increase
                assignin('base','RepTraining',  repTrainingForTesting); % PENDING: revisar si esto hace algo

                assignin('base','rangeDown', rangeDown_testing); 
                assignin('base','emgRepetition', rangeDown_testing);


                [testing_summary_episodes, testing_summary_classifications_mode,...
                    testing_wins_by_episode, testing_loses_by_episode] = ...
                    qnn.test(params.window_size, params.stride, userData, ...
                    orientation, verbose_level-1, repTrainingForTesting,false);


                recognition_accuracy_testing(epoch, index_id_user) = testing_summary_episodes(1)/(testing_summary_episodes(1) + testing_summary_episodes(2));
                classif_mode_ok_accuracy_testing(epoch, index_id_user) = testing_summary_classifications_mode(1)/(testing_summary_classifications_mode(1)+testing_summary_classifications_mode(2));
                classif_by_window_accuracy_testing(epoch, index_id_user) = sum(testing_wins_by_episode)/(sum(testing_wins_by_episode)+sum(testing_loses_by_episode));

                accuracy_by_episode_testing(epoch, :, index_id_user) = testing_wins_by_episode ./ (testing_wins_by_episode + testing_loses_by_episode);


            end
            
            

        end
        
        
    else
    
        % TRAINING
        % for each user in Specific

        assignin('base','RepTraining',  RepTraining);

        for index_id_user=1:num_users
            user_folder = "user"+list_users(index_id_user);

            userData = loadSpecificUserByName(user_folder);
            index_in_packet = getUserIndexInPacket(dataPacket, user_folder);

            assignin('base', 'userIndex', index_in_packet);
            assignin('base','rangeDown', rangeDown);
            assignin('base','emgRepetition', rangeDown);

            [summary_episodes, summary_classifications_mode, wins_by_episode, ...
                loses_by_episode] = qnn.train(params.window_size, ...
                                    params.stride, userData, orientation, ...
                                    verbose_level-1, RepTraining);


            recognition_accuracy_training(epoch, index_id_user) = summary_episodes(1)/(summary_episodes(1) + summary_episodes(2));
            classif_mode_ok_accuracy_training(epoch, index_id_user) = summary_classifications_mode(1)/(summary_classifications_mode(1)+summary_classifications_mode(2));
            classif_by_window_accuracy_training(epoch, index_id_user) = sum(wins_by_episode)/(sum(wins_by_episode)+sum(loses_by_episode));

            accuracy_by_episode_training(epoch, :, index_id_user) = wins_by_episode ./ (wins_by_episode + loses_by_episode);

        end



        if make_validation
            % VALIDATION
            % The validation is made with theta trained with all users
            % and for each epoch. It should increase
            assignin('base','RepTraining',  repTrainingForValidation); % PENDING: revisar si esto hace algo

            for index_id_user=1:num_users
                user_folder = "user"+list_users(index_id_user);

                userData = loadSpecificUserByName(user_folder);
                index_in_packet = getUserIndexInPacket(dataPacket, user_folder);

                assignin('base', 'userIndex', index_in_packet);

                assignin('base','rangeDown', rangeDown_validation); 
                assignin('base','emgRepetition', rangeDown_validation);



                [validation_summary_episodes, validation_summary_classifications_mode,...
                    validation_wins_by_episode, validation_loses_by_episode] = ...
                    qnn.test(params.window_size, params.stride, userData, ...
                    orientation, verbose_level-1, repTrainingForValidation, true);


                recognition_accuracy_validation(epoch, index_id_user) = validation_summary_episodes(1)/(validation_summary_episodes(1) + validation_summary_episodes(2));
                classif_mode_ok_accuracy_validation(epoch, index_id_user) = validation_summary_classifications_mode(1)/(validation_summary_classifications_mode(1)+validation_summary_classifications_mode(2));
                classif_by_window_accuracy_validation(epoch, index_id_user) = sum(validation_wins_by_episode)/(sum(validation_wins_by_episode)+sum(validation_loses_by_episode));

                accuracy_by_episode_validation(epoch, :, index_id_user) = validation_wins_by_episode ./ (validation_wins_by_episode + validation_loses_by_episode);

            end
        end

        if make_testing

            % TESTING
            % The test is made with theta trained with all users
            % and for each epoch. It should increase
            assignin('base','RepTraining',  repTrainingForTesting); % PENDING: revisar si esto hace algo

            for index_id_user=1:num_users
                user_folder = "user"+list_users(index_id_user);

                userData = loadSpecificUserByName(user_folder);
                index_in_packet = getUserIndexInPacket(dataPacket, user_folder);

                assignin('base', 'userIndex', index_in_packet);

                assignin('base','rangeDown', rangeDown_testing); 
                assignin('base','emgRepetition', rangeDown_testing);


                [testing_summary_episodes, testing_summary_classifications_mode,...
                    testing_wins_by_episode, testing_loses_by_episode] = ...
                    qnn.test(params.window_size, params.stride, userData, ...
                    orientation, verbose_level-1, repTrainingForTesting,false);


                recognition_accuracy_testing(epoch, index_id_user) = testing_summary_episodes(1)/(testing_summary_episodes(1) + testing_summary_episodes(2));
                classif_mode_ok_accuracy_testing(epoch, index_id_user) = testing_summary_classifications_mode(1)/(testing_summary_classifications_mode(1)+testing_summary_classifications_mode(2));
                classif_by_window_accuracy_testing(epoch, index_id_user) = sum(testing_wins_by_episode)/(sum(testing_wins_by_episode)+sum(testing_loses_by_episode));

                accuracy_by_episode_testing(epoch, :, index_id_user) = testing_wins_by_episode ./ (testing_wins_by_episode + testing_loses_by_episode);

            end

        end
        
    end
end

header_row = cell(1, num_users+1);
header_row{1,1} = "Epoch\\USER";
column_epochs = reshape(1:params.numEpochs, params.numEpochs, 1);

for index_id_user=1:num_users
    header_row{1, index_id_user+1} = "user"+list_users(index_id_user);
end

writeExperimentsAccuracyExcel(excel_dir, "TRAINING", ...
    header_row, column_epochs, recognition_accuracy_training, ...
    classif_mode_ok_accuracy_training, classif_by_window_accuracy_training)


if make_validation
    writeExperimentsAccuracyExcel(excel_dir, "VALIDATION", ...
        header_row, column_epochs, recognition_accuracy_validation, ...
        classif_mode_ok_accuracy_validation, classif_by_window_accuracy_validation)
end

if make_testing
    writeExperimentsAccuracyExcel(excel_dir, "TESTING", ...
        header_row, column_epochs, recognition_accuracy_testing, ...
        classif_mode_ok_accuracy_testing, classif_by_window_accuracy_testing)
end

elapsedTimeHours = toc(t_start)/3600;


% TODO: verificar que el accuracy calculado es apropiado
% I only take the result from all users in the last epoch for final accuracy
training_accuracy = (mean(recognition_accuracy_training(end, :)) + mean(classif_mode_ok_accuracy_training(end, :)) + mean(classif_by_window_accuracy_training(end, :)))/3;

if make_validation
    validation_accuracy = (mean(recognition_accuracy_validation(end, :)) + mean(classif_mode_ok_accuracy_validation(end, :)) + mean(classif_by_window_accuracy_validation(end, :)))/3;
end

if make_testing
    testing_accuracy = (mean(recognition_accuracy_testing(end, :)) + mean(classif_mode_ok_accuracy_testing(end, :)) + mean(classif_by_window_accuracy_testing(end, :)))/3;
end

if verbose_level >= 1
    fprintf("Elapsed time: %2.2f h", elapsedTimeHours);
    fprintf("************************\nSummary Parameters, results\n************************\n");

    disp(params.numNeuronsLayers);
    disp(params.transferFunctions);
    fprintf("%.2f %d %d %d %d %.2f %.2f\n", ...
        params.learningRate, ...
        params.window_size, params.stride, params.miniBatchSize, params.reserved_space_for_gesture, ...
        params.epsilon, params.gamma);
    
    fprintf("\nMean window accuracy Training:%2.2f, Validation:%2.2f, Testing:%2.2f\n", ...
        mean(classif_by_window_accuracy_training(end, :)), ...
        mean(classif_by_window_accuracy_validation(end, :)), ... 
        mean(classif_by_window_accuracy_testing(end, :)));
   
end


if params.numEpochs > 1
    saveFiguresSummaryByEpochAndUser(dir_specific_experiment, filename_experiment, ...
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

end

showFigureAccuracyByEpisode(dir_specific_experiment, filename_experiment, ...
        list_users, ...
        accuracy_by_episode_training, ...
        accuracy_by_episode_validation, ...
        accuracy_by_episode_testing, ...
        RepTraining, repTrainingForValidation, repTrainingForTesting, ...
        make_validation, make_testing, 2)

figure(3);
plot(1:length(qnn.cost), qnn.cost);
xlabel('N° update weights')
ylabel('Cost')
legend('training cost')
title('Cost by update')

saveas(gcf, dir_specific_experiment + "/Figure_" + filename_experiment +"_costUpdateTheta.png");

close all;

end

