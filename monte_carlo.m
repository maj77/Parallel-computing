%% inits
clc; clear; close all;

iterations = 100;
left_b = 0.0;
right_b = 1.5*pi;

%% actual monte carlo
func_val = 0;
sum = 0;
for n=1:1:iterations
    rand_num = left_b + rand(1) * (right_b-left_b);
    func_val = sin(rand_num);
    sum = sum + func_val;
end
result = (right_b-left_b)*sum/iterations;