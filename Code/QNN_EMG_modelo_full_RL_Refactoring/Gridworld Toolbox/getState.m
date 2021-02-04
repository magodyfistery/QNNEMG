function state = getState(state, action)
% Getting position of the agent
[rowAgent, colAgent] = find(state(:, :, 4) == 1);
currentPosition = [rowAgent, colAgent];
% Getting position of the wall
[rowWall, colWall] = find(state(:, :, 3) == 1);
wall = [rowWall, colWall];

% Getting the new state
state(:, :, 4) = 0;
% Up
if action == 1
    newRowAgent = rowAgent - 1;
    newColAgent = colAgent;
    newPosition = [newRowAgent, newColAgent];
    if newPosition(1) >= 1 && ~isequal(newPosition, wall)
        state(newPosition(1), newPosition(2), 4) = 1;
    else
        state(currentPosition(1), currentPosition(2), 4) = 1;
    end
end
% Down
if action == 2
    newRowAgent = rowAgent + 1;
    newColAgent = colAgent;
    newPosition = [newRowAgent, newColAgent];
    if newPosition(1) <= size(state, 1) && ~isequal(newPosition, wall)
        state(newPosition(1), newPosition(2), 4) = 1;
    else
        state(currentPosition(1), currentPosition(2), 4) = 1;
    end
end
% Left
if action == 3
    newRowAgent = rowAgent;
    newColAgent = colAgent - 1;
    newPosition = [newRowAgent, newColAgent];
    if newPosition(2) >= 1 && ~isequal(newPosition, wall)
        state(newPosition(1), newPosition(2), 4) = 1;
    else
        state(currentPosition(1), currentPosition(2), 4) = 1;
    end
end
% Right
if action == 4
    newRowAgent = rowAgent;
    newColAgent = colAgent + 1;
    newPosition = [newRowAgent, newColAgent];
    if newPosition(2) <= size(state, 2) && ~isequal(newPosition, wall)
        state(newPosition(1), newPosition(2), 4) = 1;
    else
        state(currentPosition(1), currentPosition(2), 4) = 1;
    end
end
end