clear;
% Model coefficients
a0_val = 0.00376903;
a1_val = 0.0731193;
a2_val = 0.469961;

b0_val = 0.0050882;

alpha1_val = 0.42;
alpha2_val = 0.19;
alpha3_val = 1.5;
alpha4_val = 0.95;

% Symbolic equations
syms x1 x2 x3 u a0 a1 a2 b0 alpha1 alpha2 alpha3 alpha4
dx1dt = -a2*x1 + x2;
dx2dt = -a1*x1 + x3;
dx3dt = -a0*x1 + b0*(alpha1*u+alpha2*u^2+alpha3*u^3+alpha4*u^4);

% Linearization point
syms ud

% Nonlinear elements
nonlinear = [u, u^2, u^3, u^4];
linearized = nonlinear;
for i=2:4
    taylor_expansion = subs(nonlinear(i), u, ud) + subs(diff(nonlinear(i)), u, ud)*(u-ud);
    linearized(i) = taylor_expansion;
end

for i=4:-1:2
    dx3dt = subs(dx3dt, nonlinear(i), linearized(i));
end

y_lin = x1;

dx1dt_lat = latex(dx1dt);
dx2dt_lat = latex(dx2dt);
dx3dt_lat = latex(dx3dt);
y_lin_lat = latex(x1);
