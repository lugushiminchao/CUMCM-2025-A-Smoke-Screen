%% 2025国赛A题 问题1：有效遮蔽时长（MATLAB向量化）
% 用法: matlab -batch "cd('.../问题1'); q1_matlab"

function q1_matlab()
clc;
thisDir = fileparts(mfilename('fullpath'));
outDir = fullfile(thisDir, '结果');
if ~exist(outDir, 'dir'), mkdir(outDir); end

fprintf('==== 问题1 MATLAB 计算 ====\n');

g = 9.8;
v_drone = 120;
v_missile = 300;
v_sink = 3;
R = 10;
t_drop = 1.5;
tau = 3.6;
t_det = t_drop + tau;
T_life = 20;

M0 = [20000; 0; 2000];
FY0 = [17800; 0; 1800];
T = [0; 200; 5];   % 真目标几何中心

dist_M0 = norm(M0);
u_m = -M0 / dist_M0;
v_m = v_missile * u_m;
t_hit = dist_M0 / v_missile;

v_fy = [-v_drone; 0; 0];
P_drop = FY0 + v_fy * t_drop;
P_det = P_drop + [v_fy(1)*tau; v_fy(2)*tau; -0.5*g*tau^2];

fprintf('|M0|=%.9f, t_hit=%.9f\n', dist_M0, t_hit);
fprintf('P_drop = [%.1f, %.1f, %.1f]\n', P_drop);
fprintf('t_det=%.1f, P_det=[%.3f, %.3f, %.3f]\n', t_det, P_det);

Ns = [5e4, 2e5, 2e6];
for k = 1:numel(Ns)
    n = Ns(k);
    t0 = tic;
    [dur, t_in, t_out] = local_shield(M0, v_m, P_det, t_det, v_sink, T, R, t_hit, T_life, n, false);
    el = toc(t0);
    fprintf('n=%9d: 时长=%.6f s, 区间[%.6f, %.6f], 耗时=%.3f ms\n', ...
        n, dur, t_in, t_out, el*1000);
end

t0 = tic;
[dur, t_in, t_out] = local_shield(M0, v_m, P_det, t_det, v_sink, T, R, t_hit, T_life, 2e5, true);
el = toc(t0);
fprintf('\n[推荐] n=200000 + 二分边界: 时长=%.6f s, [%.6f, %.6f], 耗时=%.3f ms\n', ...
    dur, t_in, t_out, el*1000);

outFile = fullfile(outDir, 'q1_matlab_result.txt');
fid = fopen(outFile, 'w', 'n', 'UTF-8');
fprintf(fid, '有效遮蔽时长 = %.6f s\n', dur);
fprintf(fid, '有效区间 = [%.6f, %.6f] s\n', t_in, t_out);
fprintf(fid, '投放点 = [%.1f, %.1f, %.1f]\n', P_drop);
fprintf(fid, '起爆点 = [%.3f, %.3f, %.3f]\n', P_det);
fprintf(fid, '起爆时刻 = %.1f s\n', t_det);
fclose(fid);
fprintf('结果已写入: %s\n', outFile);
end


function [dur, t_in, t_out] = local_shield(M0, v_m, P_det, t_det, v_sink, T, R, t_hit, T_life, n, refine)
t0 = t_det;
t1 = min(t_det + T_life, t_hit);
ts = linspace(t0, t1, n);
[d, s] = local_los(ts, M0, v_m, P_det, t_det, v_sink, T);
ok = (d <= R) & (s >= 0) & (s <= 1);
dt = ts(2) - ts(1);
dur = sum(ok) * dt;

idx = find(diff(ok) ~= 0);
if isempty(idx)
    if any(ok)
        t_in = t0; t_out = t1;
    else
        t_in = NaN; t_out = NaN;
    end
    return;
end
edges = zeros(size(idx));
for i = 1:numel(idx)
    j = idx(i);
    lo = ts(j); hi = ts(j+1);
    left_ok = ok(j);
    if refine
        for it = 1:50
            mid = 0.5*(lo+hi);
            [dm, sm] = local_los(mid, M0, v_m, P_det, t_det, v_sink, T);
            mid_ok = (dm <= R) && (sm >= 0) && (sm <= 1);
            if mid_ok == left_ok
                lo = mid;
            else
                hi = mid;
            end
        end
    end
    edges(i) = 0.5*(lo+hi);
end
t_in = edges(1);
t_out = edges(min(2, numel(edges)));
if refine && numel(edges) >= 2
    dur = t_out - t_in;
end
end


function [d, s] = local_los(ts, M0, v_m, P_det, t_det, v_sink, T)
ts = ts(:);
Mx = M0(1) + v_m(1)*ts;
My = M0(2) + v_m(2)*ts;
Mz = M0(3) + v_m(3)*ts;
Cx = P_det(1) * ones(size(ts));
Cy = P_det(2) * ones(size(ts));
Cz = P_det(3) - v_sink*(ts - t_det);
ABx = T(1) - Mx; ABy = T(2) - My; ABz = T(3) - Mz;
ACx = Cx - Mx;   ACy = Cy - My;   ACz = Cz - Mz;
L2 = ABx.^2 + ABy.^2 + ABz.^2;
s = (ACx.*ABx + ACy.*ABy + ACz.*ABz) ./ max(L2, 1e-18);
sc = min(max(s, 0), 1);
px = Mx + sc.*ABx; py = My + sc.*ABy; pz = Mz + sc.*ABz;
d = sqrt((Cx-px).^2 + (Cy-py).^2 + (Cz-pz).^2);
end
