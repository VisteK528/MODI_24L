clear;
model_setup

set(0, 'defaulttextinterpreter','latex');
set(0, 'DefaultLineLineWidth',1);
set(0, 'DefaultStairLineWidth',1);
resolution = 400;

t_probing = [0.1, 1, 3];
steps = ["malego", "sredniego", "duzego"];
u_min = 0;
u_max = 0.5;


for j=1:3
        T = t_probing(j);
        simout_nonlin = sim("continuous_model.slx",'Solver','ode45','StartTime','0','StopTime', int2str(t_fin));
        simout_discrete = sim("discrete_model_nonlinear.slx",'Solver','ode45','StartTime','0','StopTime', int2str(t_fin));
        y_nonlin = simout_nonlin.get("y_non_linear").Data;
        t_nonlin = simout_nonlin.get("t_non_linear").Data;
        u_nonlin = simout_nonlin.get("u_non_linear").Data;
        
        y_discrete = simout_discrete.get("y_discrete").Data;
        t_discrete = simout_discrete.get("y_discrete").time;
    
        % Plot the characteristics
        file_name = "images/ex8_simulations_t=" + T + ".png";
        figure;

        % Left Yaxis plot
        plot(t_nonlin, y_nonlin, "--");
        hold on;
        stairs(t_discrete(1:length(y_discrete)), y_discrete);
        hold on;
        
        % Left Yaxis setup
        padding_left = 0.1; 
        yLimits_left = ylim(gca);
        newMin_left = yLimits_left(1) - padding_left*(yLimits_left(2)-yLimits_left(1));
        newMax_left = yLimits_left(2) + padding_left*(yLimits_left(2)-yLimits_left(1));
        ylim(gca, [newMin_left, newMax_left]);
        
        xlabel('$t(s)$', 'fontsize', 18);
        ylabel('$y$', 'fontsize', 18);

        labels = get(gca,'YTickLabel');
        labels = strrep(labels (:),'.',',');
        set(gca,'YTickLabel',labels);
        labels = get(gca,'XTickLabel');
        labels = strrep(labels (:),'.',',');
        set(gca,'XTickLabel',labels);

         % Right Yaxis plot
        yyaxis right;
        plot(t_nonlin, u_nonlin, '-.');

        % Right Yaxis setup
        padding_right = 0.1;
        yLimits_right = ylim(gca);
        newMin_right = yLimits_right(1) - padding_right*(yLimits_right(2)-yLimits_right(1));
        newMax_right = yLimits_right(2) + padding_right*(yLimits_right(2)-yLimits_right(1));
        ylim(gca, [newMin_right, newMax_right]);
        hold off;

        ylabel('$u$', 'fontsize', 18);
        labels = get(gca,'YTickLabel');
        labels = strrep(labels (:),'.',',');
        set(gca,'YTickLabel',labels);

        
        % General export setup
        discrete_label = "Model nieliniowy dyskretny: T="+T;
        discrete_label = strrep(discrete_label (:),'.',',');

        x0=10;
        y0=10;
        width=1280;
        height=720;
        set(gcf,'position',[x0,y0,width,height]);
        grid(gca,'minor');
        legend("Model nieliniowy ciąły", discrete_label, "Sygnał sterujący",'Location','best', 'fontsize', 12)
        exportgraphics(gcf, file_name, 'Resolution', resolution);
 end
