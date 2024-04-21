% Continuous model
syms x1 x2 x3 u a0 a1 a2 b0 alpha1 alpha2 alpha3 alpha4
dx1dt = -a2*x1 + x2;
dx2dt = -a1*x1 + x3;
dx3dt = -a0*x1 + b0*(alpha1*u+alpha2*u^2+alpha3*u^3+alpha4*u^4);

dXdt = [dx1dt dx2dt dx3dt];
X = [x1 x2 x3];
syms x1k_next x2k_next x3k_next x1k x2k x3k T
Xk_next = [x1k_next x2k_next x3k_next];
Xk = [x1k x2k x3k];

% Discretization 
for i =1:3
    Xk_next(i) = solve (dXdt(i) == (Xk_next(i) - Xk(i)) / T, Xk_next(i));
    for j =1:3
        Xk_next(i) = subs(Xk_next(i), X(j) ,Xk(j) ) ;
    end
    Xk_next(i) = collect(expand(Xk_next(i)));
end
yk = Xk(1);
