function writeExperimentsAccuracyExcel(full_dir, sheet_name, ...
    header_row, column_epochs, recognition_accuracy, ...
    classif_mode_ok_accuracy, classif_by_window_accuracy)
%WRITEEXPERIMENTSACCURACYEXCEL Summary of this function goes here
%   Detailed explanation goes here

row_position = 1;
offset = numel(column_epochs) + 3;

should_write = true;
while should_write
    try 
        writematrix(column_epochs, full_dir, 'Sheet',sheet_name,'Range',"A"+(row_position+2));
        writematrix(recognition_accuracy, full_dir, 'Sheet',sheet_name,'Range',"B"+(row_position+2));
        writetable(cell2table({"RECOGNITION ACCURACY"}), full_dir,'Sheet',sheet_name,'Range',"A"+row_position,'WriteVariableNames',false);
        writetable(cell2table(header_row), full_dir,'Sheet',sheet_name,'Range',"A"+(row_position+1),'WriteVariableNames',false);

        should_write = false;

    catch ME 
        disp(ME.identifier); 
        disp(ME.message); 
        disp("Reintentando en 1 a 3 segundo"); 
        pause(randi(3)); 
    end     
end


row_position = row_position + offset;

should_write = true;
while should_write
    try 
        writematrix(column_epochs, full_dir, 'Sheet',sheet_name,'Range',"A"+(row_position+2));
        writematrix(classif_mode_ok_accuracy, full_dir, 'Sheet',sheet_name,'Range',"B"+(row_position+2));
        writetable(cell2table({"CLASIFICATION ACCURACY"}), full_dir,'Sheet',sheet_name,'Range',"A"+row_position,'WriteVariableNames',false);
        writetable(cell2table(header_row), full_dir,'Sheet',sheet_name,'Range',"A"+(row_position+1),'WriteVariableNames',false);

        should_write = false;

    catch ME 
        disp(ME.identifier); 
        disp(ME.message); 
        disp("Reintentando en 1 a 3 segundo"); 
        pause(randi(3)); 
    end     
end

row_position = row_position + offset;
should_write = true;
while should_write
    try 
        writematrix(column_epochs, full_dir, 'Sheet',sheet_name,'Range',"A"+(row_position+2));
        writematrix(classif_by_window_accuracy, full_dir, 'Sheet',sheet_name,'Range',"B"+(row_position+2));
        writetable(cell2table({"WINDOW ACCURACY"}), full_dir,'Sheet',sheet_name,'Range',"A"+row_position,'WriteVariableNames',false);
        writetable(cell2table(header_row), full_dir,'Sheet',sheet_name,'Range',"A"+(row_position+1),'WriteVariableNames',false);

        should_write = false;

    catch ME 
        disp(ME.identifier); 
        disp(ME.message); 
        disp("Reintentando en 1 a 3 segundo"); 
        pause(randi(3)); 
    end     
end
        
        

end

