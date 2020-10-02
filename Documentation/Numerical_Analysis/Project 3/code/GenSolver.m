% Input as vector
function GenSolver(m,c,k,initial)

% Initializing the value in M matrix
m1 = m(1);
m2 = m(2);
m3 = m(3);
m4 = m(4);
  
% Initializing the value in C matrix
c1 = c(1);
c2 = c(2);
c3 = c(3);
  
% Initializing the value in K matrix
k1 = k(1);
k2 = k(2);
k3 = k(3);
k4 = k(4);
k5 = k(5);

% Initializing M matrix
M = diag([m1 m2 m3 m4]);

% Initializing C matrix
C = [c1+c2 0 -c2 0; 0 0 0 0; -c2 0 c2+c3 -c3; 0 0 -c3 c3];

% Initializing K matrix
K = [k1+k2+k5 -k2 -k5 0; -k2 k2+k3 -k3 0; -k5 -k3 k3+k4+k5 -k4; 0 0 -k4 k4];

% Computing eigenpairs with polynomial
[eigVec, eigVal, CondNum] = polyeig(K, C, M);

% Setting up the linear system
A = zeros(size(initial,1), 2);
B = [initial(1); initial(2); initial(3); initial(4)];
coeff = [];

% Calculating the coefficient using Ax = b
% A is the real and imaginary value of eigenvector
% B is the initial values 
for i = 1: 2: size(eigVal)
    A(:,1) = real(eigVec(:,i));
    A(:,2) = imag(eigVec(:,i));
    sol = linsolve(A,B);
    
    % Concatinating coefficients
    coeff = [coeff; sol];
end

% Plotting
t = linspace(0,10);
y1 = exp(real(eigVal(1) * t));
y2 = coeff(1,1) * (real(eigVec(1,1)) * cos(imag(eigVal(1) * t)) - imag(eigVec(1,1)) * sin(imag(eigVal(1) * t)));
y3 = coeff(2,1) * (real(eigVec(1,1)) * sin(imag(eigVal(1) * t)) + imag(eigVec(1,1)) * cos(imag(eigVal(1) * t)));
% Elements to elements multiplicatio2
y = times(y1, (y2 + y3))

figure
plot(y)
end