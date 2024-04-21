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

X = solve([dx1dt, dx2dt, dx3dt] == zeros(3,1), [x1; x2; x3]);
y = X.x1;
y_latex = latex(y);

% Symbolic variables substitution
y_subs = subs(y, {a0, a1, a2, b0, alpha1, alpha2, alpha3, alpha4}, {a0_val, a1_val, a2_val, b0_val, alpha1_val, alpha2_val, alpha3_val, alpha4_val});

% Range of steering signal
u_range = linspace(-1, 1);

y_exp = double(subs(y_subs, u, u_range));

% Plot the characteristics
file_name = 'images/ex1_nonlinear_static_characteristics.png';
title_str = "Nieliniowa charakterystyka statyczna $y(u)$";
resolution = 600;

set(0, 'defaulttextinterpreter','latex');
set(0, 'DefaultLineLineWidth',1);
set(0, 'DefaultStairLineWidth',1);

figure;
plot(u_range, y_exp);
ylim([-1.2, 5]);
xlabel('$u$', 'fontsize', 18);
ylabel('$y$', 'fontsize', 18);
x0=10;
y0=10;
width=1280;
height=720;
set(gcf,'position',[x0,y0,width,height]);
grid(gca,'minor');
% title(title_str, 'FontSize', 22);

labels = get(gca,'YTickLabel');
labels = strrep(labels (:),'.',',');
set(gca,'YTickLabel',labels);
labels = get(gca,'XTickLabel');
labels = strrep(labels (:),'.',',');
set(gca,'XTickLabel',labels);


exportgraphics(gcf, file_name, 'Resolution', resolution);



