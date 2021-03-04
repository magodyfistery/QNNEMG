function displayAction(state, action, varargin)
if nargin == 2
    flagColorBar = true;
else
    flagColorBar = varargin{1};
end
displayWorld(state, flagColorBar);
hold on;
% Getting position of the agent
[rowAgent, colAgent] = find(state(:, :, 4) == 1);
y = rowAgent + 0.5;
x = colAgent + 0.5;
arrowLength = 0.4;
% Up
if action == 1
    u = 0;
    v = -arrowLength;
end
% Down
if action == 2
    u = 0;
    v = arrowLength;
end
% Left
if action == 3
    u = -arrowLength;
    v = 0;
end
% Right
if action == 4
    u = arrowLength;
    v = 0;
end
quiver(x, y, u, v, 'Linewidth', 1, 'Color', [0, 0, 0], 'MaxHeadSize', 2);
drawnow;
end
