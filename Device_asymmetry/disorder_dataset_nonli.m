% order = randperm(n_train);
load data/train_order.mat
order = order;
train_x = train_x(order, :);
train_y = train_y(order, :);
fprintf('Already disordered the training dataset\n');

