% Init parameters
continuous_model_setup

linearized = false;

% Non-linear static characteristic
u = linspace(-1, 1);

non_linear_model = @(u)((b0)/(a0))*(alpha1*u + alpha2*u.^2 + alpha3*u.^3 + alpha4*u.^4);
y_non_linear = non_linear_model(u);


if linearized
    linearized_model = @(u, lin_point)((b0)/(a0))*(alpha1*u + alpha2*(lin_point^2 + 2*lin_point*(u-lin_point)) + alpha3*(lin_point^3 + 3*lin_point^2*(u-lin_point)) + alpha4*(lin_point^4+4*lin_point^3*(u-lin_point)));
    y_linearized = linearized_model(u, lin_point);
end

set(0 ,'defaulttextinterpreter','latex');
set(0, 'DefaultLineLineWidth', 1);

figure;
plot(u, y_non_linear);

if linearized
    hold on;
    plot(u, y_linearized);
    xline(lin_point, '--');
    yline(non_linear_model(lin_point), '--');
    title_string = "Charakterystyka zlinearyzowana dla $\overline{u}=" + lin_point + "$ na tle nieliniowej charakterystyki statycznej $y(u)$";
    title(title_string, 'FontSize', 22);
else
    title("Nieliniowa charakterystyka statyczna $y(u)$", 'FontSize', 22);
end




ylim([-1.2, 5]);
xlabel('$u$', 'fontsize', 18);
ylabel('$y$', 'fontsize', 18);
x0=10;
y0=10;
width=1280;
height=720;
set(gcf,'position',[x0,y0,width,height])
exportgraphics(gcf, 'rys.png', 'Resolution', 600);