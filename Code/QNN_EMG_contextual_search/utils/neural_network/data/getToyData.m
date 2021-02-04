function [X y] = getToyData()
  
  
  %{
  X = [0 0; 0 1; 1 0; 1 1; 1 0; 1 1; 1 0; 0 0; 0 1; 1 0; 1 1;];
  y = [1; 1; 1; 2; 1; 2; 1; 1; 1; 1; 2];
  %}
  
  X = [0 0; 0 1; 1 0; 1 1; 1 0; 1 1; 1 0; 0 0; 0 1; 1 0; 1 1];
  y = [1; 0; 0; 1; 0; 1; 0; 1; 0; 0; 1] + 1;
  
  
  %{
  samples = 100;
  features = 10;

  X = ones(samples, features);
  y = ones(samples, 1);

  for i=1:samples
    direction = randi([-1, 1]);
    X(i, :) = (1:features) * direction;
    y(i, 1) = direction + 2;
  endfor
  
  %}
end
