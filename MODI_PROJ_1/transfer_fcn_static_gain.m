% Coefficients
a0_val = 0.00376903;
a1_val = 0.0731193;
a2_val = 0.469961;

b0_val = 0.0050882;

alpha1_val = 0.42;
alpha2_val = 0.19;
alpha3_val = 1.5;
alpha4_val = 0.95;

% Linearization points
ud_1 = 0.05;
ud_2 = 0.5;
ud_3 = 0.85;

syms a0 a1 a2 b0 alpha1 alpha2 alpha3 alpha4 s ud
A = [-a2 1 0; -a1 0 1; -a0 0 0];
B = [0; 0; b0*(4*alpha4*ud^3 + 3*alpha3*ud^2 + 2*alpha2*ud + alpha1)];
C = [1 0 0];
D = 0;

% Transfer function
G = C*inv(s*eye(3, 3) - A)*B + D;


% Static gain
G_lim = subs(G, s, 0);

K_stat_1 = double(subs(G_lim, {a0, a1, a2, b0, alpha1, alpha2, alpha3, alpha4, ud}, {a0_val, a1_val, a2_val, b0_val, alpha1_val, alpha2_val, alpha3_val, alpha4_val, ud_1}));
K_stat_2 = double(subs(G_lim, {a0, a1, a2, b0, alpha1, alpha2, alpha3, alpha4, ud}, {a0_val, a1_val, a2_val, b0_val, alpha1_val, alpha2_val, alpha3_val, alpha4_val, ud_2}));
K_stat_3 = double(subs(G_lim, {a0, a1, a2, b0, alpha1, alpha2, alpha3, alpha4, ud}, {a0_val, a1_val, a2_val, b0_val, alpha1_val, alpha2_val, alpha3_val, alpha4_val, ud_3}));

