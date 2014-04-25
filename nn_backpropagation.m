function nn = nn_backpropagation(nn)
	levels = nn.levels;
	da_fun = nn.derivative_activation_fun;

    args.i = levels;
    d{levels} = - nn.e .* da_fun(nn, args);
    
	for i = (levels - 1) : -1 : 2
        args.i = i;
		d_a = da_fun(nn, args);
        
        if i+1 == levels
            d{i} = (d{i + 1} * nn.W{i}) .* d_a;
        else
            d{i} = (d{i + 1}(:,2:end) * nn.W{i}) .* d_a;
        end
    end
	
	for i = 1 : (levels - 1)
        if i+1==levels
            nn.dW{i} = (d{i + 1}' * nn.a{i}) / size(d{i + 1}, 1);
        else
            nn.dW{i} = (d{i + 1}(:,2:end)' * nn.a{i}) / size(d{i + 1}, 1);      
        end
    end

end
	
