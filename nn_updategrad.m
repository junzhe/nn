function nn = nn_updategrad(nn)
	for i = 1 : nn.levels - 1
		dW = nn.dW{i};

		dW = nn.learningrate * nn.dW{i};

		if(nn.momentum>0)
            		nn.vW{i} = nn.momentum*nn.vW{i} + dW;
            		dW = nn.vW{i};
        	end

		nn.W{i} = nn.W{i} - dW;
	end
end
