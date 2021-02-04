function displayWorld(state, varargin)
if nargin == 1
    flagColorBar = true;
else
    flagColorBar = varargin{1};
end

% Draws the current state of the game
world = zeros(size(state, 1), size(state, 2));
% Agent
world = max(world, 4*state(:, :, 4));
% Wall
world = max(world, 3*state(:, :, 3));
% Pit
world = max(world, 2*state(:, :, 2));
% Goal
world = max(world, 1*state(:, :, 1));

padded = padarray(world, [1,1], 'post');
% Color map
map = [0.5, 0.5, 0.5;... % color of the world
    0.0, 1.0, 0.1;...  % color of the goal
    1.0, 0.0, 0.1;...  % color of the pit
    0.5, 0.2, 0.07;... % color of the wall
    1.0, 1.0, 0.0];  % color of the agent
pcolor(padded);
colormap(map);
if flagColorBar
    colorbar('Ticks', [0.4, 1.2, 2.0, 2.8, 3.6],...
        'TickLabels',{'Ground','Goal','Pit','Wall','Agent'});
end
axis equal tight;
ax = gca;
ax.XTick = 1.5:1:4.5;
ax.XTickLabels = 1:1:4;
ax.YTick = 1.5:1:4.5;
ax.YTickLabels = 1:1:4;
axis ij;
drawnow;
end