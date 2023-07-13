%% RESET p
% b1_fit = load('.\data\b1_fit_3.dat');  
load('.\data_1\b1_fit_5.mat');
b1_fit_nonli = b1_fit_5;

c1_fit = load('.\data\c1_fit.dat'); 
c1_fit = c1_fit/5  /1.8;

%% SET p
% a2 = load('a2_fit.dat');
% b2_fit = load('.\data\b2_fit_3.dat');
load('.\data_1\b2_fit_5.mat');
b2_fit_nonli = b2_fit_5;

% b2_fit(1:3) = b2_fit(1:3)/2;
% b2_fit(4:6) = b2_fit(4:6)./ (3:5);
% b2_fit(7:10) = b2_fit(7:10)./ (5:8);
% b2_fit = b2_fit  /1.75;

c2_fit = load('.\data\c2_fit.dat');
c2_fit = c2_fit/3.3  /1.8;

%% initial
b_RS = [b1_fit_nonli; zeros(1,size(b1_fit_nonli,2)); b2_fit_nonli]; 
% c_RS = [c1_fit; zeros(1,size(c1_fit,2)); c2_fit];  
c_RS = [zeros(1,size(c1_fit,2)); zeros(1,size(c1_fit,2)); zeros(1,size(c1_fit,2))];  