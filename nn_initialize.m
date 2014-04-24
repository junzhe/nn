function nn = nn_initialize(opts)
	nn.layers = opts.layers;
	nn.levels = numel(nn.layers);
	nn.learningrate = opts.learningrate;
	nn.lambda = opts.lambda;

	for i = 2 : nn.levels
		nn.W{i - 1} = (rand(nn.size(i), nn.size(i - 1)+1) - 0.5) * 2 * 4 * sqrt(6 / (nn.size(i) + nn.size(i - 1)));
        	nn.vW{i - 1} = zeros(size(nn.W{i - 1}));
	end
end
