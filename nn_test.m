function [er, bad] = nn_test(nn, x, y)
    labels = nn_predict(nn, x);
    [~, expected] = max(y,[],2);
    bad = find(labels ~= expected);    
    er = numel(bad) / size(x, 1);
end
