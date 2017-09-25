load('a1a.mat');
y = label*0.5+0.5;
y = y';


% A=[A ones(1605,1)];
H = A.'*A/1605;
v = ones(119,1);
% v(120) = 0;
% v = (0:0.1:11.8)';
Hv = H * v

grad = -A'*y/1605;
% grad(end) = -sum(y)/1605;
x = zeros(119,1);
% x = CG(H, grad, x);

