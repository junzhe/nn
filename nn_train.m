function [nn, L]  = nn_train(nn, train_x, train_y, opts, val_x, val_y)
	m = size(train_x, 1);

	batchsize = opts.batchsize;
	numepochs = opts.numepochs;

	numbatches = m / batchsize;

	assert(rem(numbatches, 1) == 0, 'numbatches must be a integer');

	L = zeros(numepochs*numbatches,1);
	n = 1;
	for i = 1 : numepochs
    
   	 	kk = randperm(m);
    		for l = 1 : numbatches
        		batch_x = train_x(kk((l - 1) * batchsize + 1 : l * batchsize), :);
        
        		batch_y = train_y(kk((l - 1) * batchsize + 1 : l * batchsize), :);
        
        		nn = nn_forwardfeed(nn, batch_x, batch_y);
        		nn = nn_backpropagation(nn);
        		nn = nn_updategrad(nn);
        
        		L(n) = nn.L;
        
        		n = n + 1;
    		end
    

    		nn.learningRate = nn.learningRate * nn.scaling_learningRate;
	end
end

