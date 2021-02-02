function new_begin_row = write_experiment_row(experiment_parameters, SHEET_NAME, row_position, summary, experiment_id, USER_ID)

new_begin_row = row_position;
% SHEET_NAME = "TESTING";
global index_numNeuronsLayers_input
global index_numNeuronsLayers_output
global index_RepTraining
global index_SamplingType  % 0=Random package, NO USADO
global index_learningRate
global index_numEpochsToIncreaseMomentum
global index_miniBatchSize
global index_lambda
global filename_experimentsQNN
filename_experimentsQNN = "../experiments/experimentsQNNtesting_best.xlsx"

countWins = summary(1);
number_classif_ok = summary(2);
countWins2 = summary(3);
countLoses = summary(4);
number_classif_failed = summary(5);
countLoses2 = summary(6);

numNeuronsLayers1 = experiment_parameters(experiment_id, index_numNeuronsLayers_input);
numNeuronsLayers2 = experiment_parameters(experiment_id, index_numNeuronsLayers_input+1);
numNeuronsLayers3 = experiment_parameters(experiment_id, index_numNeuronsLayers_input+2);
numNeuronsLayers4 = experiment_parameters(experiment_id, index_numNeuronsLayers_output);
RepTraining = experiment_parameters(experiment_id, index_RepTraining);
samplingType = "Unknown";
if experiment_parameters(experiment_id, index_SamplingType) == 0
    samplingType = "Random Package";
end
learningRate = experiment_parameters(experiment_id, index_learningRate);
numEpochsToIncreaseMomentum = experiment_parameters(experiment_id, index_numEpochsToIncreaseMomentum);
miniBatchSize = experiment_parameters(experiment_id, index_miniBatchSize);
lambda = experiment_parameters(experiment_id, index_lambda); 

table_row = table(experiment_id, numNeuronsLayers1, numNeuronsLayers2, ...
    numNeuronsLayers3, numNeuronsLayers4, RepTraining, samplingType, ...
    learningRate, numEpochsToIncreaseMomentum, miniBatchSize, lambda, ...
    countWins,number_classif_ok,countWins2,countLoses, ...
    number_classif_failed,countLoses2, ...
	(round(countWins/(countWins+countLoses), 4)*100)+"%", ...
	(round(number_classif_ok/(number_classif_ok+number_classif_failed), 4)*100)+"%");

table_row.Properties.VariableNames = {'Experience' 'numNeuronsLayers1' ...
    'numNeuronsLayers2' 'numNeuronsLayers3' 'numNeuronsLayers4' ...
    'RepTraining'	'Sampling type' ...
    'learningRate' 'numEpochsToIncreaseMomentum' 'miniBatchSize' 'lambda' ...
    'Count Wins Recog' 'Count Wins Classif' 'Count Wins per window' ...
    'Count Loses Recog'	'Count Loses Classif' 'Count Loses per window' ...
	'Recognition' 'Classification'};


should_write = true; 

while should_write 
	new_begin_row = row_position;
	try 
		writecell({USER_ID}, filename_experimentsQNN,'Sheet',SHEET_NAME,'Range',"A"+new_begin_row);
		
		writetable(table_row, filename_experimentsQNN,'Sheet', SHEET_NAME,'Range', "B"+new_begin_row,'WriteVariableNames',false);
		new_begin_row = new_begin_row + 1;

		should_write = false; 

	catch ME 

		disp(ME.identifier); 

		disp(ME.message); 

		disp("Reintentando en 1 a 3 segundo"); 

		pause(randi(3)); 

	end 

  

end



end

