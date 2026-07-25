function [P_drop, P_det, t_det, v_fy] = strategy_kinematics(theta, v, t_drop, tau, p)
%STRATEGY_KINEMATICS  由决策变量计算投放点/起爆点
%   theta  : 水平航向角（相对 +x，弧度），等高度飞行
%   v      : 无人机速度 m/s
%   t_drop : 投放时刻 s（受领任务后）
%   tau    : 引信延时 s（投放→起爆）
%   p      : scenario_params()

v_fy = [v*cos(theta); v*sin(theta); 0];
P_drop = p.FY0 + v_fy * t_drop;
t_det  = t_drop + tau;
% 弹体：水平速度继承，竖直自由落体
P_det = P_drop + [v_fy(1)*tau; v_fy(2)*tau; -0.5*p.g*tau^2];
end
