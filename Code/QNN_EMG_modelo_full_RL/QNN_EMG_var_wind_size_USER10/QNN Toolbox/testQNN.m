function result = testQNN(weights, transferFunctions, options, typeWorld, typeControl)
% Creating a new instance of the world
state = createWorld(typeWorld);
if strcmp(typeControl, 'AI')
    displayWorld(state);
else
    displayWorld(state, false);
end
gameOn = true;
maxNumSteps = 10;
stepNum = 0;
while gameOn
    stepNum = stepNum + 1;
    if strcmp(typeControl, 'AI')
        [dummyVar, A] = forwardPropagation(state(:)', weights,...
            transferFunctions, options);
        Qval = A{end}(:, 2:end);
        [dummyVar, action] = max(Qval);
    else
        % Read the action from keyboard
        keyPressed = getkey();
        % up = 30, down = 31, left = 28, right = 29
        if keyPressed == 30, action = 1; end
        if keyPressed == 31, action = 2; end
        if keyPressed == 28, action = 3; end
        if keyPressed == 29, action = 4; end
    end
    % Taking the selected action
    if strcmp(typeControl, 'AI')
        displayAction(state, action);
    else
        displayAction(state, action, false);
    end
    new_state = getState(state, action);
    reward = getReward(new_state);
    if strcmp(typeControl, 'AI')
        displayWorld(new_state);
    else
        displayWorld(new_state, false);
    end
    title(['Trial = ' num2str(stepNum) ' of ' num2str(maxNumSteps)]);
    pause(0.10);
    if reward == +10
        result = 1;
        title('Game over: Your agent WON :)');
        drawnow;
        break;
    elseif reward == -10
        result = 0;
        title('Game over: Your agent LOST :(');
        drawnow;
        break;
    end
    if stepNum >= maxNumSteps
        result = 0;
        title('Game over: Your agent LOST :(');
        drawnow;
        break;
    end
    state = new_state;
end
end
