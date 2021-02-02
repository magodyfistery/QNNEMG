@echo off
:: este script es para facilitar el trabajo de copiar carpetas de especific con datos de full data
:: copia la carpeta base y agrega usuarios desde %1 hasta %2

::PARÁMETROS
::	%1 -> usuario inicial
::	%2 -> hasta usuario final
:: Ejemplo de uso "0_replicate_with_user_data.bat 80 306"

:: Se recolecta parámetros desde consola
SET /A inicio = %1
SET /A fin = %2 + 1

:: este es el directorio de la carpeta a copiar
set folder=QNN_EMG_var_wind_size_USER

:: este es el directorio relativo donde están todos los datos SIN EL NUMERO FINAL
set data_folder_user=../../Data_full/Specific/user


:: For usuario=%1 hasta usuario=%2, en orden secuencial.

:begin
IF %inicio% == %fin% GOTO end
:: =====================

:: nombre de la carpeta a copiarse
set new_folder=%folder%%inicio%

:: copia la carpeta original y le da el nombre anteriormente declarado (agrega el número del usuario
robocopy "%folder%" "%new_folder%" /MIR

:: Se agrega el número al dato del usuario user+1=user1 ...../Specific/user%2
set data_user=%data_folder_user%%inicio%

:: Copia el dato del usuario en la carpeta que se copió antes
set folder_specific=%new_folder%/Data/Specific/user%inicio%
robocopy "%data_user%" "%folder_specific%" /MIR

:: Aumenta el contador del ciclo while
:: =====================
SET /A inicio = %inicio% + 1
GOTO begin

:prohibidouso
echo NO EJECUTAR

:end
echo FINALIZADO
