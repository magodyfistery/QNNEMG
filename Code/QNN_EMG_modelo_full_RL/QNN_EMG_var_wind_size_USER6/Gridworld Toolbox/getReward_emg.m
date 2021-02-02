function reward = getReward_emg(action, etiqueta_actual)
% % Getting position of the agent
% [rowAgent, colAgent] = find(state(:, :, 4) == 1);
% currentPosition = [rowAgent, colAgent];
% % Getting position of the pit
% [rowPit, colPit] = find(state(:, :, 2) == 1);
% pit = [rowPit, colPit];
% % Getting position of the goal
% [rowGoal, colGoal] = find(state(:, :, 1) == 1);
% goal = [rowGoal, colGoal];

% Getting the reward
% if isequal(currentPosition, goal)
%     reward = +10; %10
% elseif isequal(currentPosition, pit)
%     reward = -10;
% else
%     reward = 0;  % -1
% end
% end


%TENGO QUE CREAR RECOMPENSA EN BASE A LA ACCION OBTENIDA POR EPSILON GREEDY
%Y AL GROUND TRUTH QUE TENGO EN BASE A MI DATASET (ESE ES MI AMBIENTE)

if action==etiqueta_actual
    reward = +1; %10
else
    reward = -1; %10
end

end