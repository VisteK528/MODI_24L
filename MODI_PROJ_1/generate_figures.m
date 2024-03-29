% Dynamic characteristics

t_nl = t_non_linear.Data;
t_l = t_linearized.Data;
t_d = t_discrete.Data;

y_non_linear_data = y_non_linear.Data;
y_linear_data = y_linearized.Data;
y_discrete_data = y_discrete.Data;

simout_dys = sim("discrete_model_nonlinear.slx",'Solver','ode45','StartTime','0','StopTime', int2str(t_fin));
y_discrete_data = simout_dys.get("y_discrete").Data;
t_d = simout_dys.get("t_discrete").Data;

figure;
plot(t_nl, y_non_linear_data);
hold on;
plot(t_l, y_linear_data);
hold on;
stairs(t_d(1:length(t_d)-2), y_discrete_data);

exportgraphics(gcf, 'rys.png', 'Resolution', 800);
