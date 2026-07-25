function p = scenario_params()
%SCENARIO_PARAMS  2025国赛A题 公共场景参数（问题1/2）
%   p = scenario_params() 返回结构体，含导弹、无人机、云团、目标等常量。

p.g = 9.8;                 % 重力加速度 m/s^2
p.v_missile = 300;         % 导弹速度 m/s
p.v_sink = 3;              % 云团下沉速度 m/s
p.R = 10;                  % 有效遮蔽半径 m
p.T_life = 20;             % 云团有效寿命 s
p.v_uav_min = 70;          % 无人机速度下界
p.v_uav_max = 140;         % 无人机速度上界

% 位置（假目标为原点）
p.M0  = [20000; 0; 2000];  % 导弹 M1 初位置
p.FY0 = [17800; 0; 1800];  % 无人机 FY1 初位置
p.Tgt = [0; 200; 5];       % 真目标几何中心（推荐遮蔽判据）

% 导弹运动（直指假目标）
p.dist_M0 = norm(p.M0);
p.u_m = -p.M0 / p.dist_M0;
p.v_m = p.v_missile * p.u_m;
p.t_hit = p.dist_M0 / p.v_missile;  % 到达假目标时刻
end
