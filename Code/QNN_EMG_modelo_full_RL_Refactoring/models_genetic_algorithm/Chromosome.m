classdef Chromosome
    
    properties
        gens = [];
        
    end
    
    methods
        function obj = Chromosome(gens_set, num_gens)
            % constructor, gens: alpha, neurons_hidden_layer1
            % gens_set is matrix, each row are 10 examples
            obj.gens = zeros(1, num_gens);
            for i=1:num_gens
                obj.gens(i) = Chromosome.choose_random_sample(gens_set(i, :), 1);
            end
        end
    end
    
    methods(Static)
       function random_sample = choose_random_sample(array, size)
            random_sample = zeros(1, size);
            for i=1:size
                choice = randi([1 length(array)]);
                random_sample(i) = array(choice);
            end
       end
    end
    
    
end

