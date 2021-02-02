@echo off
:: este script es para facilitar el trabajo de ejecutar los scripts de cada carpeta

:: PREREQUISITOS:
::	haber ejecutado el codigo 0 para un número de usuarios en un rango mayor que el que se usará aquí o no encontrará la carpeta

:: ejecuta la carpeta base con un numero por delante

::PARÁMETROS
::	%1 -> numEpochs (veces que se muestra los datos)
::	%2 -> window_sizest en vector dentro de string  "[100 105]"
::	%3 -> stridest en vector dentro de string  "[20 30 40]"
::	%4 -> usuario inicial
::	%5 -> hasta usuario final


:: Ejemplo de uso en el CMD:
:: 1_execute_matlab_instruction_for_folder.bat 1 "[100 105]" "[20 30 40]" 1 30

:: para ejecutar un usuario por usuario se usa de inicio 1 hasta 1, 4 a 4, 306 a 306, etc.
:: Ejemplo de solo ejecutar el usuario 288: 1_execute_matlab_instruction_for_folder.bat 1 "[200]" "[20 30 40]" 288 288

SET /A numEpochs = %1

set window_sizest=%2
set stridest=%3
set window_sizes=%window_sizest:~1,-1%
set strides=%stridest:~1,-1%

SET /A inicio = %4
SET /A fin = %5 + 1

:: este es el directorio BASE (sin número) de la carpeta en donde está el escript
set folder=QNN_EMG_var_wind_size_USER


:: For usuario=%1 hasta usuario=%2, en orden secuencial.
:begin
IF %inicio% == %fin% GOTO end
:: =====================

set execute_from_folder=%folder%%inicio%

:: al abrir una instancia de matlab por consola solo se puede ejecutar una instrucción en una cadena de texto "", puede comporse de varias llamadas a varias cosas de matlab siempre y cuando se separe con punto y coma

:: cd('C:\Users\ALAN_TURING\Desktop\AUTOMATED Code QNN\QNN_wind-stride_improved\%execute_from_folder%'); QNN_train_emg_Exp_Replay_SxWx(%inicio%, 27, 32, true, true, [250], [20 40 50]);  quit;


set instruction="try; actual_dir=pwd; cd(actual_dir+string('/%execute_from_folder%'));  QNN_train_emg_Exp_Replay_SxWx(%inicio%, 27, 32, true, true, %window_sizes%, %strides%, %numEpochs%); quit; catch ME; disp(ME); end;"

echo %instruction%

:: matlab -nosplash -noFigureWindows -r 
matlab -nosplash -noFigureWindows -r %instruction%
timeout 15

:: =====================
SET /A inicio = %inicio% + 1
GOTO begin





:prohibidouso
echo NO EJECUTAR

:end
echo FINALIZADO
