function [cosDistance] = cosDisFunc(vector1,vector2)
%UNTITLED 此处提供此函数的摘要
%   此处提供详细说明
vector1 = squeeze(vector1);
vector2 = squeeze(vector2);
cosDistance = dot(vector1,vector2) / (norm(vector1)*norm(vector2));


end