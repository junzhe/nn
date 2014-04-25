clear all;
close all;

train_x = [];
train_y = [];

load data_batch_1
train_x = [train_x; double(data)/255];
train_y = zeros(length(labels), 10);

for i = 1:length(labels)
    train_y(i, labels(i)+1) = 1;
end

load data_batch_2
test_x = double(data)/255;
test_y = zeros(length(labels), 10);

for i = 1:length(labels)
    test_y(i, labels(i)+1) = 1;
end

opts.layers = [32*32*3, 1024, 512, 10];
opts.activation_fun = @nn_sigm;
opts.derivative_activation_fun = @nn_der_sigm;
opts.learningrate = 2;
opts.scaling_learningrate = 1;
opts.lambda = 3e-3;
opts.momentum = 0.5;

nn = nn_initialize(opts);

opts.numepochs = 30;
opts.batchsize = 100;

[nn, L] = nn_train(nn, train_x, train_y, opts);

err = nn_test(nn, test_x, test_y);

err