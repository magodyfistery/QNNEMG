classdef Population < handle
    
    properties(Constant)
        THRESHOLD_FITNESS = 95;
    end
    
    properties
        % vector of the object Chromosome
        chromosomes = []; 
        gens_set = [];
        max_population = 0;
        mutation_rate = 0.9;
        
        fitness_best = -1;
        fitness_avg = -1;
    end
    
    methods
        function this = Population(max_population, gens_set, mutation_rate)
            % set of posible gens for population
            this.gens_set = gens_set;
            % max population     
            this.max_population = max_population;
            % mutation percentage
            this.mutation_rate = mutation_rate;
            
            
            
        end
        
        function generate_initial_population(this, num_gens)
            % initial population
            this.chromosomes = Chromosome.empty([0 this.max_population]);
            for i=1:this.max_population
                this.chromosomes(i) = Chromosome(this.gens_set, num_gens);
            end
        end
        
        function fitness = fitness_function(this, verbose_level)
            % target is the unicode target for this specific string problem
            
            fitness = zeros(1, this.max_population);
            
            for i=1:this.max_population
                
                if verbose_level >= 1
                    fprintf("Individual %d of %d\n", i, this.max_population);
                end
                params.learningRate = this.chromosomes(i).gens(1);
                params.neurons_hidden1 = ceil(this.chromosomes(i).gens(2));
                params.neurons_hidden2 = ceil(this.chromosomes(i).gens(3));
                params.miniBatchSize = ceil(this.chromosomes(i).gens(4));
                window_size = ceil(this.chromosomes(i).gens(5));
                stride = ceil(this.chromosomes(i).gens(6));
                params.reserved_space_for_gesture = ceil(this.chromosomes(i).gens(7));
                params.epsilon = this.chromosomes(i).gens(8);
                % not implemented or not relevant to optimize
                params.lambda = 0;
                params.gamma = 1;
                params.initialMomentum = 0.3;
                params.momentum = 0.9;
                params.numEpochsToIncreaseMomentum = 50;
                params.typeWorld = 'randWorld';
                params.W = 25;
                params.rewardType = 1;
                
                [training_accuracy, test_accuracy, ~] = QNN_emg_Exp_Replay(..., 
                params, window_size, stride, "genetic_individual", -1, 87, 13);
                
                % ponderation of the accuracys
                % penalization = abs(training_accuracy - test_accuracy); % penalizate overfitting 
                penalization = 0;
                fitness(i) = (training_accuracy+test_accuracy)/2 - penalization;
            end
        end
        
        function [offspring_mutated, new_register_change] = mutate(this, offspring_crossover, reductor_factor, register_change)

            offspring_mutated = offspring_crossover;
            % Mutation changes a single gene in each offspring randomly.
            num_individuals = length(offspring_mutated);
            num_gens = length(offspring_mutated(1).gens);

            poles = [-1, 1];

            for individual=1:num_individuals
                % The random value to be added to the gene.

                if rand() <= this.mutation_rate
                    
                    random_gen = randi([1 num_gens]);
                    minimum_valid_number = this.gens_set(random_gen, 1);
                    maximum_valid_number = this.gens_set(random_gen, end);
                    
                    change = mean([minimum_valid_number maximum_valid_number])/(reductor_factor * 2); 
                    random_value = change * poles(randi([1 2]));
                    
                    register_change{random_gen} = [register_change{random_gen} change];  % change is absolute positive
                    
                    
                    offspring_mutated(individual).gens(random_gen) = ...
                        clamp(minimum_valid_number, maximum_valid_number, ...
                              offspring_mutated(individual).gens(random_gen) + ...
                              random_value);
                end
            end
            
            new_register_change = register_change;


        end
        
        
    end
    methods(Static)
        function best_individuals_index = select_best_individuals_by_elitist(fitness, num_best_individuals)
           % elitist method only select the two best parents 
           % Selecting the best individuals in the current generation as 
            % parents for producing the offspring of the next generation.

            % very low numbers
            best_individuals_values = ones(1, num_best_individuals)*-Inf;
            best_individuals_index = zeros(1, num_best_individuals);

            for i=1:numel(fitness)
                individual_fitness = fitness(i);

                [first_min_value, first_min_index] = min(best_individuals_values);

                if individual_fitness > first_min_value
                    best_individuals_values(first_min_index) = individual_fitness;
                    best_individuals_index(first_min_index) = i;
                end

            end
        end
        
        function offspring = crossover(parents, num_individuals)
            % IT CAN BE IMPROVED. Only making combinations of 2 parents

            num_parents = numel(parents);
            num_gens = length(parents(1).gens);
            
            offspring = Chromosome.empty([0 num_individuals]);
            
            % The point at which crossover takes place between two parents. 
            % Usually, it is at the center. There are a lot of methods
            % better than this one.
            
            crossover_point = floor(num_gens/2)+1;  % +1 is for MATLAB array index starts in 1.

            % its a secuencial crossover, which is bad due to prob. duplication
            for individual=0:num_individuals-1
                % Index of the first parent to mate. +1 due to MATLAB index
                parent1_idx = mod(individual, num_parents)+1;
                % Index of the second parent to mate. +1 due to MATLAB index
                parent2_idx = mod(individual+1, num_parents)+1;
                % The new offspring will have its first half of its genes taken from
                % the first parent. +1 due to MATLAB index
                offspring(individual+1) = parents(parent1_idx);
                % The new offspring will have its second half of its genes taken from
                % the second parent. +1 due to MATLAB index
                offspring(individual+1).gens(crossover_point:end) = parents(parent2_idx).gens(crossover_point:end);

            end
        end
        
        function bool_value = hasReachedTheTop(fitness)
            
            mean_fitness = mean(fitness);
            bool_value = mean_fitness >= Population.THRESHOLD_FITNESS;
        end
        
        
       
             
    end
end

