function userData = loadSpecificUserByName(folder_name)
%LOADSPECIFICUSER Summary of this function goes here
%   Detailed explanation goes here

pathUser        = pwd;  % the site where the app is running
pathOrigin      = 'Data\Specific';

path_user_data=(horzcat(pathUser,'\',pathOrigin,'\',char(folder_name),'\','userData.mat'));
myload = load(path_user_data);
userData = myload.userData;

end

