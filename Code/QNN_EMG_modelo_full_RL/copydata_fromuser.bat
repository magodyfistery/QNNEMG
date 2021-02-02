@echo off
:: este script es para facilitar el trabajo de copiad
:: %1 -> inicio, %2 -> final
:: Ejemplo de uso "Nombre.bat 80 306"

:: SEGURO -> comentar si se quiere usar el script
:: GOTO prohibidouso

SET /A inicio = %1
SET /A fin = %2 + 1

:: cd  "C:\Git\QNN_emg\CodigoFuente\QNN_modelo\QNN_EMG_RandomData_6ok - USER"



set folder=C:\Users\ALAN_TURING\Desktop\AUTOMATED Code QNN\QNN_wind-stride\QNN_EMG_var_wind_size_USER
set dir_csv=results\csv_full_test_data
set dir_figures=results\figures




:begin
IF %inicio% == %fin% GOTO end
:: =====================
:: cd  "%folder%%inicio%"

set source_csv=%folder%%inicio%/%dir_csv%
set source_figures=%folder%%inicio%/%dir_figures%

set destination_csv=results/user%inicio%/csv
set destination_figures=results/user%inicio%/figures

:: echo %source%
:: echo %destination%
robocopy "%source_csv%" "%destination_csv%" /MIR
robocopy "%source_figures%" "%destination_figures%" /MIR

:: cd ..
:: =====================
SET /A inicio = %inicio% + 1

GOTO begin




:prohibidouso
echo NO EJECUTAR

:end
echo FINALIZADO
