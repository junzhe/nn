function nn = nn_initialize(opts)
	nn.layers = opts.layers;
	nn.levels = numel(nn.layers);
	nn.learningrate = opts.learningrate;
    nn.scaling_learningrate = opts.scaling_learningrate;
	nn.lambda = opts.lambda;
    nn.activation_fun = opts.activation_fun;
    nn.derivative_activation_fun = opts.derivative_activation_fun;
    nn.momentum = opts.momentum;
    
	for i = 2 : nn.levels
		nn.W{i - 1} = (rand(nn.layers(i), nn.layers(i - 1)+1) - 0.5) * 2 * 4 * sqrt(6 / (nn.layers(i) + nn.layers(i - 1)));
        nn.vW{i - 1} = zeros(size(nn.W{i - 1}));
	end
end
