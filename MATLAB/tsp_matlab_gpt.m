clear; close all;

d = [0, 1.3, 3.9, 1.7, 3.0; 1.3, 0, 2.4, 4.5, 5.7; 3.9, 2.4, 0, 1.2, 6.1; 1.7, 4.5, 2.5, 0, 2.2; 3.0, 5.7, 6.1, 2.2, 0];
N = 8;
d = rand(N,N);
for n=1:N
    d(n,n) = 0;
end
[cost, path] = tsp_viterbi_trellis(d, true, N);