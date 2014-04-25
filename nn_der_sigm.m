function [ X ] = nn_der_sigm(nn, args)
    i = args.i;
    a = nn.a{i};
    
    X = a .* (1 - a);
end

