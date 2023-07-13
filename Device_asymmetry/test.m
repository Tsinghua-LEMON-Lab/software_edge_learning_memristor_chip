function [y1, y2, acc] = test(input, w1, w2, target)

y1 = max(input*w1, 0);
y2 = y1*w2;

[~, pred] = max(y2, [], 2);
acc = sum(pred == target) / length(target);

end

