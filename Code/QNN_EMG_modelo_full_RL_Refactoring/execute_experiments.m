function execute_experiments(model_name, ...
    verbose_level, list_users, make_validation, make_testing, user_reset, experiment_ids)

dir_results = 'Experiments/resultsQNNByExperiment/'+model_name;

[status, msg, msgID] = mkdir(dir_results);
if ~isempty(msgID)
    fprintf("\n\n%s. Status: %d. ESE MODELO YA EXISTE, pulse ENTER para continuar SOBREESCRIBIENDO\n\n", msg, status);
    pause;
end

addpath('QNN Toolbox');
addpath(genpath('utils'))
addpath('Experiments')

number_experiments = numel(experiment_ids);

parameters = table2array(readtable('Experiments/experiments_parameters_QNN.csv'));

header = {'Experience';	'repTrainingTrain'; ...
    'repTrainingValidation'; 'repTrainingTest'; 'window_size'; 'stride'; ...
    'numNeuronsInput'; 'numNeuronsHidden1'; 'numNeuronsHidden2'; ...
    'numNeuronsOutput';	'transferFunctionHidden1'; 'transferFunctionHidden2'; ...
	'transferFunctionOutput'; 'initialMomentum'; 'momentum'; ...
    'numEpochsToIncreaseMomentum'; ...
    'learningRate'; 'lambda'; 'rewardType'; 'gamma'; 'reserved_space_for_gesture'; ...
    'miniBatchSize'; 'epsilon'; 'numEpochs'};

summary_experiments = zeros(number_experiments, 3);
column_experiments = zeros(number_experiments,1);

for index_experiment_ids=1:number_experiments
    experiment_id = experiment_ids(index_experiment_ids);
    column_experiments(index_experiment_ids, 1) = experiment_id;
    % only get the row with a set of params
    params_experiment_row = parameters(parameters(:, 1) == experiment_id, :);
    
    
    
    repTrainingTrain = params_experiment_row(2);
    repTrainingValidation = params_experiment_row(3);
    repTrainingTesting = params_experiment_row(4);
    
    params = build_params(params_experiment_row);
    
    filename_experiment = "experimentId_" + experiment_id;
    dir_specific_experiment = dir_results + "/"+filename_experiment;
    [~, ~, ~] = mkdir(dir_specific_experiment);
    
    if params.numEpochs > 1
        [~, ~, ~] = mkdir(dir_specific_experiment + "/Figures_summary_byUserEpochs");
    end
    
    excel_dir = dir_specific_experiment + "/results_" + filename_experiment + ".xlsx";
    writetable(cell2table(header), excel_dir,'Sheet',"PARAMS",'Range',"A1",'WriteVariableNames',false);
    writematrix(params_experiment_row', excel_dir, 'Sheet',"PARAMS",'Range',"B1");

    
    
    [training_accuracy, validation_accuracy, testing_accuracy, qnn] = ...
        QNN_emg_Exp_Replay(params, verbose_level, repTrainingTrain, ...
        repTrainingValidation, repTrainingTesting, ...
         dir_specific_experiment, filename_experiment, excel_dir, list_users, ...
        make_validation, make_testing, user_reset);
    
    if verbose_level >= 1
        fprintf("Final Ponderation.- Train:%2.2f Validation:%2.2f Test:%2.2f\n", training_accuracy, validation_accuracy, testing_accuracy);
        
    end
    
    summary_experiments(index_experiment_ids, 1) = training_accuracy;
    summary_experiments(index_experiment_ids, 2) = validation_accuracy;
    summary_experiments(index_experiment_ids, 3) = testing_accuracy;
    
    model_dir = dir_specific_experiment + "/model.mat";
    theta = qnn.theta;
    gameReplay = qnn.gameReplay;
    save(model_dir,'theta', 'gameReplay');


    
end

header_final = {'Experiment\\Accuracy' 'Training' 'Validation' 'Testing'};


writetable(cell2table(header_final), dir_results+"/final_summary.xlsx",'Sheet',"PARAMS",'Range',"A1",'WriteVariableNames',false);
writematrix(column_experiments, dir_results+"/final_summary.xlsx", 'Sheet',"PARAMS",'Range',"A2");
writematrix(summary_experiments, dir_results+"/final_summary.xlsx", 'Sheet',"PARAMS",'Range',"B2");


end