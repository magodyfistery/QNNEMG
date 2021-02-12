function transfer_function = castNumberToTransferFunction(id_transfer_function)
%CASTNUMBERTOTRANSFERFUNCTION Summary of this function goes here
%   Detailed explanation goes here

if id_transfer_function == 1
    transfer_function = 'relu';
elseif id_transfer_function == 2
    transfer_function = 'purelin';
elseif id_transfer_function == 3
    transfer_function = 'sigmoid';
else
    disp('Transfer function not supported');
    transfer_function = 'relu';
end

end

