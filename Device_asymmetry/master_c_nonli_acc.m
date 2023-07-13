% clear
% close all
% clc
tic

load('data\model_dict_9776.mat');
load('data\mnist_single.mat')  
n_train = length(train_x); 
n_test = length(test_y); 
n_class = max(train_y);
disorder_dataset_nonli  

train_y_onehot = single(zeros(n_class, n_train));  
temp = 0:n_class:(n_train*n_class - 1);  
train_y_onehot(train_y + temp') = 10;  
train_y_onehot = train_y_onehot';  

load_model_paramter_nonli  

grad_data = zeros(9,30000);
weight_updated = zeros(9,100,10);
weight_updated_bp = zeros(9,100,10);
acc_list = zeros(9,1);
pulse_num_list = zeros(9,1);

for threshold = 0:8
    fc1 = rram_array(0, 0, 0, fc1_weight, 0, 0, 0, 0); 
    fc2 = rram_array(fc2_weight_pos, fc2_weight_neg, fc2_weight_G, fc2_weight, fc2_weight_range, fc2_weight_G_scale, b_RS, c_RS);

    rand_data_name = 'data\rand_data0.mat';
    fc2 = fc2.randomized_weight(rand_data_name);  % ！！！！！！注意这句话不能注释！！！！！！

    epoch = 5;  
    batch_size = 1;
    max_iter = epoch * n_train / batch_size;   
    
    
    lr = 5e-3;  
%     noise = 0.03;  
    noise = 0;
    pulse_bound = 50;  

    best_acc = 0;

    for i=1:max_iter

        start_idx = mod((i-1) * batch_size, n_train) + 1;  
        end_idx = start_idx + batch_size - 1;
        input = train_x(start_idx:end_idx, :); 
        label_onehot = train_y_onehot(start_idx:end_idx, :);
      
        y1 = max(input*fc1.weight, 0);
        y2 = y1*fc2.weight;
        
%         fc2 = fc2.bp_update(y1, y2, label_onehot, lr, noise);
%         fc2 = fc2.bp_update_th(y1, y2, label_onehot, lr, noise, threshold);
        fc2 = fc2.circuit_sbp_update_single_update_wosz(y1, y2, label_onehot, threshold);

    
        
        if mod(i, 5000) == 0
            [~, ~, acc] = test(test_x, fc1.weight, fc2.weight, test_y);
    
            if acc > best_acc
                best_acc = acc;
            end
        end

    end
    fprintf('======== Done ========\n');  
    fprintf('best_acc: %6.4f\n', best_acc); 
    toc

    fprintf('fc2 update pulse num bp w verify : %d\n', fc2.pulse_num);
    acc_list(threshold+1,:) = best_acc;
    pulse_num_list(threshold+1,:) = fc2.pulse_num;

end


