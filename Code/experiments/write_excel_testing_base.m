function new_begin_row = write_excel_testing_base(filename_experimentsQNN, SHEET_NAME)

row_position = 1;

header_part1 = {'Experience' 'numNeuronsLayers'};
header_part2 = {'RepTraining'	'Sampling type' ...
    'learningRate' 'numEpochsToIncreaseMomentum' 'miniBatchSize' 'lambda' ...
    'Count Wins Recog' 'Count Wins Classif' 'Count Wins per window' ...
    'Count Loses Recog'	'Count Loses Classif' 'Count Loses per window' ...
	'Recognition' 'Classification'};



should_write = true; 

while should_write 
	new_begin_row = row_position;
	try 
		%%% Creating sheet in excel
		writecell({"QNN - EMG - 5 gestures - Experience Replay"}, filename_experimentsQNN,'Sheet',SHEET_NAME,'Range',"A"+new_begin_row);
		
		writecell({"TESTING RESULTS"}, filename_experimentsQNN,'Sheet',SHEET_NAME,'Range',"N"+new_begin_row);
		new_begin_row = new_begin_row + 1;
		writecell({"User"}, filename_experimentsQNN,'Sheet',SHEET_NAME,'Range',"A"+new_begin_row);
		writetable(cell2table(header_part1), filename_experimentsQNN,'Sheet',SHEET_NAME,'Range',"B"+new_begin_row,'WriteVariableNames',false);
		writetable(cell2table(header_part2), filename_experimentsQNN,'Sheet',SHEET_NAME,'Range',"G"+new_begin_row,'WriteVariableNames',false);
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

