function labels = nnpredict(nn, x)
    nn = nn_forwardfeed(nn, x, zeros(size(x,1), nn.layers(end)));
    
    [~, i] = max(nn.a{end},[],2);
    labels = i;
end
