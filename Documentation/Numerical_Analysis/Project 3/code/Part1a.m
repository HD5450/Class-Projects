%Part a
m1 = 1;
m2 = 2;
m3 = 3;
m4 = 1;

c1 = 0.1;
c2 = 0.2;
c3 = 0.3;

k1 = 1;
k2 = 2;
k3 = 1;
k4 = 4;
k5 = 3;

M = diag([m1 m2 m3 m4]);
C = [c1+c2 0 -c2 0; 0 0 0 0; -c2 0 c2+c3 -c3; 0 0 -c3 c3];
K = [k1+k2+k5 -k2 -k5 0; -k2 k2+k3 -k3 0; -k5 -k3 k3+k4+k5 -k4; 0 0 -k4 k4];

% X's column is eigenvector
% e is eigenvalue
% s conditional number for eignevalues
[X,e,s] = polyeig(K, C, M)

% Checking eigenvector
lambda = e(1);
x = X(:,1);
% Should =0 for verification
(M*lambda^2 + C*lambda + K)*x

