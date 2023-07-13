function [max_abs_value] = max_abs(input_vector)
% 根据input vector 计算最大的绝对值
max_pos = max(input_vector);
max_neg = min(input_vector)*-1;

max_abs_value = max(max_pos, max_neg);

end