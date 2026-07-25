%% 纯计时对比脚本：与 Python 同规模
function q1_matlab_bench()
clc;
g = 9.8; v_drone = 120; v_missile = 300; v_sink = 3; R = 10;
t_drop = 1.5; tau = 3.6; t_det = t_drop + tau; T_life = 20;
M0 = [20000;0;2000]; FY0 = [17800;0;1800]; T = [0;200;5];
dist_M0 = norm(M0); u_m = -M0/dist_M0; v_m = v_missile*u_m; t_hit = dist_M0/v_missile;
v_fy = [-v_drone;0;0];
P_drop = FY0 + v_fy*t_drop;
P_det = P_drop + [v_fy(1)*tau; v_fy(2)*tau; -0.5*g*tau^2];

fprintf('==== MATLAB 计时 ====\n');
Ns = [5e4, 2e5, 2e6];
for n = Ns
    t0 = tic;
    ts = linspace(t_det, min(t_det+T_life, t_hit), n);
    Mx = M0(1)+v_m(1)*ts; My = M0(2)+v_m(2)*ts; Mz = M0(3)+v_m(3)*ts;
    Cx = P_det(1)*ones(size(ts)); Cy = P_det(2)*ones(size(ts));
    Cz = P_det(3) - v_sink*(ts - t_det);
    ABx = T(1)-Mx; ABy = T(2)-My; ABz = T(3)-Mz;
    ACx = Cx-Mx; ACy = Cy-My; ACz = Cz-Mz;
    L2 = ABx.^2+ABy.^2+ABz.^2;
    s = (ACx.*ABx+ACy.*ABy+ACz.*ABz)./max(L2,1e-18);
    sc = min(max(s,0),1);
    d = sqrt((Cx-(Mx+sc.*ABx)).^2 + (Cy-(My+sc.*ABy)).^2 + (Cz-(Mz+sc.*ABz)).^2);
    ok = (d<=R) & (s>=0) & (s<=1);
    dur = sum(ok) * (ts(2)-ts(1));
    el = toc(t0);
    fprintf('n=%9d: duration=%.6f s, time=%.3f ms\n', n, dur, el*1000);
end
end
