function state = createWorld(typeWorld)
switch typeWorld
    case 'deterministic'
        state = zeros(4, 4, 4);
        % Placing the agent
        state(4, 1, 4) = 1;
        % Placing the wall
        state(3, 2, 3) = 1;
        % Placing the pit
        state(3, 3, 2) = 1;
        % Placing the goal
        state(2, 3, 1) = 1;
    case 'randAgent'
        state = zeros(4, 4, 4);
        % Placing the wall
        state(3, 2, 3) = 1;
        % Placing the pit
        state(3, 3, 2) = 1;
        % Placing the goal
        state(2, 3, 1) = 1;
        % Placing the agent at a random location
        while true
            [dummy, idx] = sort(rand(1, 16));
            randIdx = idx(1);
            world = sum(state, 3);
            if world(randIdx) == 0
                [randRow, randCol] = ind2sub(size(world), randIdx);
                state(randRow, randCol, 4) = 1;
                break;
            end
        end
    case 'randWorld'
        state = zeros(4, 4, 4);
        % 1 = goal, 2 = pit, 3 = wall, 4 = agent
        for i = 1:4
            % Placing the elements of the world in random locations
            while true
                [dummy, idx] = sort(rand(1, 16));
                randIdx = idx(1);
                world = sum(state, 3);
                if world(randIdx) == 0
                    [randRow, randCol] = ind2sub(size(world), randIdx);
                    state(randRow, randCol, i) = 1;
                    break;
                end
            end
        end
    otherwise
        fprintf('Wrong type of world\n');
end
end
