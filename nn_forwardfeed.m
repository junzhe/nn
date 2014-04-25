function nn = nn_forwardfeed(nn, x, y)
	levels = nn.levels;
	x_len = size(x, 1);

    x = [ones(x_len, 1) x];
    nn.a{1} = x;
    
	a_fun = nn.activation_fun;
	
	for i = 2 : levels - 1
        args.i = i-1;
		nn.a{i} = a_fun(nn, args);
		
		nn.a{i} = [ones(x_len,1) nn.a{i}];
    end
    
    args.i = levels - 1;
    nn.a{levels} = a_fun(nn, args);
   
    nn.e = y - nn.a{levels};
end

