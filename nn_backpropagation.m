function nn = nn_backpragation(nn)
	levels = nn.levels;
	da_fun = nn.derivative_activation_fun;

	for i = (levels - 1) : -1 : 2
		d_a = da_fun(nn.a{i));
        	
		d{i} = (d{i + 1}(:,2:end) * nn.W{i} + sparsityError) .* d_act;
	end
	
	for i = 1 : (levels - 1)
            nn.dW{i} = (d{i + 1}(:,2:end)' * nn.a{i}) / size(d{i + 1}, 1);
    	end

end
	
