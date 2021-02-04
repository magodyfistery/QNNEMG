function reward = getReward(state)
% Getting position of the agent
[rowAgent, colAgent] = find(state(:, :, 4) == 1);
currentPosition = [rowAgent, colAgent];
% Getting position of the pit
[rowPit, colPit] = find(state(:, :, 2) == 1);
pit = [rowPit, colPit];
% Getting position of the goal
[rowGoal, colGoal] = find(state(:, :, 1) == 1);
goal = [rowGoal, colGoal];
% Getting the reward
if isequal(currentPosition, goal)
    reward = +10; %10
elseif isequal(currentPosition, pit)
    reward = -10;
else
    reward = -1;  % -1
end
end