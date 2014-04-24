function nn = nn_forwardfeed(nn, x, y)
	levels = nn.levels;
	x_len = size(x, 1);

	a_fun = nn.activation_fun;
	
	for i = 2 : levels
		nn.a{i} = a_fun(nn.a{i - 1} * nn.W{i - 1}');
		
		nn.a{i} = [ones(m,1) nn.a{i}];
	end
end

