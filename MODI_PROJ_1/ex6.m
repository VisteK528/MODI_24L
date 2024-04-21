clear;
model_setup

set(0, 'defaulttextinterpreter','latex');
set(0, 'DefaultLineLineWidth',1);
set(0, 'DefaultStairLineWidth',1);
resolution_dpi = 400;

lin_points = [0.05, 0.5, 0.85];

u_min = 0;
u_maxs = [0.1, 0.5, 0.8];
steps = ["malego", "sredniego", "duzego"];

for i=1:3
    u_max = u_maxs(i);
    for j=1:3
        lin_point = lin_points(j);
        simout_nonlin = sim("continuous_model.slx",'Solver','ode45','StartTime','0','StopTime', int2str(t_fin));
        simout_lin = sim("continuous_model_linearized.slx",'Solver','ode45','StartTime','0','StopTime', int2str(t_fin));
        y_nonlin = simout_nonlin.get("y_non_linear").Data;
        t_nonlin = simout_nonlin.get("t_non_linear").Data;
        
        y_lin = simout_lin.get("y_linearized").Data;
        t_lin = simout_lin.get("t_linearized").Data;

        u_nonlin = simout_nonlin.get("u_non_linear").Data;
        u_lin = simout_lin.get("u_linearized").Data;
    
        % Plot the characteristics
        file_name = "images/ex6_simulations_u_max="+ u_max + "u=" + lin_point + ".png";
        figure;

        % Left Yaxis plot
        plot(t_nonlin, y_nonlin);
        hold on;
        plot(t_lin, y_lin, '--');
        hold on;
        
        % Left Yaxis limits setup
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
        legend("Model nieliniowy", "Model zlinearyzowany", "Sygnał sterujący",'Location','best', 'fontsize', 12);
        x0=10;
        y0=10;
        width=1280;
        height=720;
        set(gcf,'position',[x0,y0,width,height]);
        grid(gca,'minor');
        exportgraphics(gcf, file_name, 'Resolution', resolution_dpi);
    end
end
