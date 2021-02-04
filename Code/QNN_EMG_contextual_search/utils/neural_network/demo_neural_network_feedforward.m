function [training_accuracy, test_accuracy] = ...
    demo_neural_network_feedforward(alpha, neurons_hidden_1, neurons_hidden_2, verbose)


addpath("data");
addpath("nn_diagnostic");
addpath("nn_learning");
addpath("utils/data_handle");
addpath("utils/math");
addpath("utils/plot");

lambda = 0;
epochs = 20;
neurons_per_layer = [-1 ceil(neurons_hidden_1) ceil(neurons_hidden_2) -1]; %

[X, y] = getData(1000);%  getData(100); getToyData();



[X_train, y_train, X_test, y_test, X_validation, y_validation] = split_train_test_validation(X, y, 0.7, 0.1);

m = size(X, 1);

number_features = size(X, 2);
number_clases = size(unique(y), 1);
neurons_per_layer(1) = number_features;
neurons_per_layer(end) = number_clases;

% neurons_per_layer

thetas_rolled = initWeights(neurons_per_layer, -2, 2);




[new_thetas_rolled, Js_train, Js_validation] = ...
    train(neurons_per_layer, thetas_rolled, epochs, X_train, y_train, X_validation, y_validation, alpha, lambda, verbose);

J_test = nnCostFunction(neurons_per_layer, new_thetas_rolled, X_test, y_test, lambda);

if verbose
    fprintf('Error final de test: %.2f\n', J_test);
end

figure(1);
hold on;
plot(1:size(Js_train, 2), Js_train, 'b');
plot(1:size(Js_validation, 2), Js_validation, 'r');
xlabel('iterations')
ylabel('Cost')
legend('training', 'validation')
title('Cost by iteration')

predictions_train = predict(neurons_per_layer, new_thetas_rolled, X_train);
predictions_test = predict(neurons_per_layer, new_thetas_rolled, X_test);

training_accuracy = mean(double(predictions_train == y_train)) * 100;
test_accuracy = mean(double(predictions_test == y_test)) * 100;

if verbose
    fprintf('\nTraining Set Accuracy: %f\n', training_accuracy);
    fprintf('\nTest Set Accuracy: %f\n', test_accuracy);
end


%{

figure(2);
displayImageData(X_test(100:104,:));
disp(predict(neurons_per_layer, new_thetas_rolled, X_test(100:104,:)))
% disp(y_test(100:104));

[Js_train Js_validation] = learningCurves(neurons_per_layer, thetas_rolled, X_train, y_train, X_validation, y_validation, alpha, lambda);

hold on;
plot(1:size(Js_train, 2), Js_train, 'b');
plot(1:size(Js_validation, 2), Js_validation, 'r');
xlabel('m')
ylabel('J')
legend('training', 'validation')
title('Learning Curves')

%}

% CAMBIANDO LA FORMA DE INICIALIZAR LOS DATOS CAMBIA TODO

end