function [grad, cost] = nn_cost(nn, x, y)

	cost = 0;
	levels = nn.levels;
	layers = nn.layers;

	nn_forwardfeed(nn, x, y);

	e = y - nn.a{levels};
	
	cost = 1/layers * e.^2 + nn.lamda/2 * sum(sum(nn.W.^2));
	
	grad = nn.dW;
end
		

