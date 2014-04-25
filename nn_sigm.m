function [ X ] = nn_sigm(nn, args)
    i = args.i;
    
    a = nn.a{i};
    W = nn.W{i};
    
    X = 1./(1+exp(-a*W'));
end

