function [dur, info] = objective_q2(x, p, n)
%OBJECTIVE_Q2  问题2 目标函数：有效遮蔽时长
%   x = [theta, v, t_drop, tau]
%   dur = 遮蔽时长（越大越好）
%   info 结构体含投放/起爆等明细

if nargin < 3 || isempty(n), n = 2e4; end

theta  = x(1);
v      = x(2);
t_drop = x(3);
tau    = x(4);

[P_drop, P_det, t_det, v_fy] = strategy_kinematics(theta, v, t_drop, tau, p);

% 粗过滤：起爆后已无有效窗口
if t_det >= p.t_hit
    dur = 0;
    info = pack_info(theta, v, t_drop, tau, v_fy, P_drop, P_det, t_det, 0, NaN, NaN);
    return;
end

% 起爆点过高/过低时仍计算，由几何自然惩罚
[dur, t_in, t_out] = shield_duration(P_det, t_det, p, n, false);

info = pack_info(theta, v, t_drop, tau, v_fy, P_drop, P_det, t_det, dur, t_in, t_out);
end

function info = pack_info(theta, v, t_drop, tau, v_fy, P_drop, P_det, t_det, dur, t_in, t_out)
info.theta  = theta;
info.heading_deg = mod(theta*180/pi, 360);
info.v      = v;
info.t_drop = t_drop;
info.tau    = tau;
info.t_det  = t_det;
info.v_fy   = v_fy;
info.P_drop = P_drop;
info.P_det  = P_det;
info.dur    = dur;
info.t_in   = t_in;
info.t_out  = t_out;
end
